import os
import platform
from collections import OrderedDict as odict
from collections import defaultdict as ddict
from pathlib import Path
from typing import Union, Dict, Optional, OrderedDict
from urllib.parse import urlparse

import annotated_text
import graphviz
import streamlit as st
import streamlit_authenticator as stauth

import dataset
import utils
from dataset import GLUE, Mention

auth_conf_fname = Path(__file__).parent / 'annotators.yaml'

st.set_page_config(layout="wide", page_icon='🍽️', page_title=f'CUTLER in {platform.node()}', initial_sidebar_state="collapsed")
debug = os.environ.get('DEBUG', False)
state = st.session_state


auth_conf = utils.load_conf(auth_conf_fname)
gh_enabled = utils.token and utils.repo and utils.org

unhashed = False
for user, spec in auth_conf['credentials']['usernames'].items():
    if len(spec['password']) != 60:
        unhashed = True
        spec['password'] = stauth.Hasher([spec['password']]).generate()[0]
if unhashed:
    utils.save_conf(auth_conf_fname, auth_conf)

authenticator = stauth.Authenticate(
    auth_conf['credentials'],
    auth_conf['cookie']['name'],
    auth_conf['cookie']['key'],
    auth_conf['cookie']['expiry_days'],
)

title_col, name_col = st.columns((6,2))

title_col.header("CUTLER: Coreference-Under-Transformation Labeler")
if debug:
    title_col.subheader("DEBUG mode, no annotations will be saved!")
with name_col:
    authenticator.login('Login', 'main')
    if state.authentication_status:
        st.write(f'Welcome *{state.name}* ({state.username})')
        dataset.add_annotator(state.username)
        with st.expander('Your Account'):
            authenticator.logout('Logout', 'main')
            if state.username:
                if authenticator.reset_password(state.username, 'Reset password'):
                    utils.save_conf(auth_conf_fname, auth_conf)
                    st.success('Password modified successfully')
    elif state.authentication_status is None:
        st.warning('Please enter your username and password')
    else:
        st.error('Username/password is incorrect')

if not state.authentication_status:
    st.stop()


def next_unannotated_docid():
    state.documents = dataset.read_unannotated_documents(state.username)
    return next(iter(state.documents.keys())) if len(state.documents) > 0 else None

# documents is a dict of {doc_id: data.Recipe} , needs to be reloaded every refresh
# if 'documents' not in state: 
state.documents = dataset.read_unannotated_documents(state.username)
if 'cur_docid' not in state:  # current document to annotate
    state.cur_docid = next_unannotated_docid()
if 'cur_evtnum' not in state:  # current event to annotate
    state.cur_evtnum = 0
if 'highlighted' not in state: # last "clicked" entity to highlight in the text panel 
    state.highlighted = None
# In-memory link annotations
# It's a nested dict { step_number -> {rel_number -> {entity: whole_or_part}} }
# the most inner dict is an OrderedDict mapping Span -> T/F value.
# (the "head" of the coref-chain is always at the end of the dict, see below)
# or for comments { step_number -> {rel_number(-2000) -> str} } (no list involved)
# Namely, at each event, all coref-under-identities will be collected into a list
# and keyed with the relation label (positive integers). 
# "BEING REPRESENTATIVE": In the OrderedDict, entity that starts with `RES.` are always 
# inserted at end, and thus they will become the "head" of the CUI chain.
# By keeping the relations keyed with the event, we can pop all relation annotations 
# related to an event, which is effectively an "undo" operation 
if 'links' not in state: 
    state.links = ddict(lambda: ddict(lambda: odict()))
# in-memory "observation" entities dict (eventNum -> {Span -> manual label})
# , where eventNum is the timing when the entity "seen" or "linked"
# if a value (the manual relation label) is "unused" or "output", the entity will be used for next steps
# until the entity is "used" for an event
if 'observed' not in state: 
    state.observed = {}
# state.observed keeps candidate entities at each event. 
# The latest status (namely, claimed or not) of an entity must be inferred from
# looking state.observed from the beginning
if 'latest_status' not in state:
    state.latest_status = {}
# keeping track of corefferred entities, a map from Span -> [Span], where the key is the "head" entity 
if 'coref_groups' not in state:
    state.coref_groups = ddict(list)
# tracking number of results of currently looking event of interest, will be 1 by default, only can be a positive int
if 'num_results' not in state:
    state.num_results = 1


def increase_numresults():
    state.num_results += 1


def decrease_numresults():
    state.num_results -= 1


def get_event_by_idx(idx) -> Optional[Mention]:
    if idx < 0: 
        return None
    return state.documents[state.cur_docid].events[idx]


def get_cur_event():
    return get_event_by_idx(state.cur_evtnum)


def span_to_str(e: Mention, affix_res=True):
    """
    In R2VQ, span.id is formatted as 
    * SENTENCE "::".join(recipe_id, f'{step_id:02}', f'{sent_id:02}')
    * MENTIONS "::".join([sentence.id, str(start_idx).zfill(3), "CRL"]) 
    * HIDDENS "::".join([sentence.id, "hidden", str(i).zfill(2), k, str(hi).zfill(2), "CRL"] 
    * VERBS "::".join([sentence.id, str(start_idx).zfill(3)]) 
    a full id looks like 'f-Y2HTQLYW::step01::sent01::005::CRL' for an extent food item. 
    """
    if isinstance(e, Mention) and e.type.upper() == 'FINAL':
        return e.text
    last_piece = e.id.rsplit(GLUE, 1)[1]
    if last_piece.startswith(utils.result_export_label):
        if affix_res:
            res_num = last_piece[len(utils.result_export_label):]
            head_token = f'{utils.phantom_result_prefix}{res_num}.{e.lemma}'
        else:
            head_token = e.lemma
    elif e.type == 'EVENT':
        head_token = e.lemma
    else:
        head_token = e.text

    return GLUE.join(map(str, [head_token, e.sent, e.start]))


def is_lightverb(links_dict):
    input_groups = {k for k, l in links_dict.items() if isinstance(k, int) and k > 0 and l}
    return not input_groups and len(links_dict[utils.result_ann_code]) == 1


def is_separationevent(links_dict):
    return not links_dict[utils.result_ann_code] and links_dict[-1] and links_dict[-2]


def next_step():
    thisstep_observations: Dict[Mention, Union[int, str]] = {}
    # `links` object will be later used for graph and serialization, so do not change anything 
    # just update `observed` object based on the links created at the current event
    cur_links = state.links[state.cur_evtnum]
    # "irrelevant" and "not-used" items are processed regardless of the event being "light" or not
    for label in {utils.irrelevant_ann_code, utils.notused_ann_code}:
        if label in cur_links:
            for ent in cur_links[label].keys():
                thisstep_observations[ent] = label
    if is_lightverb(cur_links):
        head_span, _ = cur_links[utils.result_ann_code].popitem()
        thisstep_observations[head_span] = utils.lightverb_export_label
        cur_links[utils.result_ann_code][head_span] = True
    else:
        for rel, ents in cur_links.items():
            if rel > utils.notused_ann_code and ents:
                head_span, _ = ents.popitem()
                thisstep_observations[head_span] = rel
                for ent, whole in ents.items():
                    thisstep_observations[ent] = utils.identity_export_label if whole else utils.meronymy_export_label
                ents[head_span] = True
    state.observed[state.cur_evtnum] = thisstep_observations
    state.latest_status = {**state.latest_status, **thisstep_observations}
    if state.cur_evtnum == len(cur_doc.events) - 1:
        state.cur_evtnum = -1
    else:
        state.cur_evtnum += 1
    state.highlighted = None
    e = get_cur_event()
    if e is not None:
        intermediate_save_name = f"{state.username}.intermediate"
        dataset.save_serialization(state.cur_docid, intermediate_save_name, f"intermediate save after {e.id} ({e.text}/{e.lemma})", serialize_to_csv())
        dataset.save_graph(state.cur_docid, intermediate_save_name, links_to_graph(True, False))
        dataset.save_dict_to_json(state.cur_docid, intermediate_save_name, state.links)


def flag_and_next_document(report_comment, gh_issue_title=None):
    if gh_issue_title is not None:
        gh_issue_body = report_comment + \
                        '\n\n---\n' + \
                        dataset.get_commented_document_text(cur_doc.id) + \
                        '\n\n---\n' + \
                        f'Document reported by {state.name} ({state.username}) via CUTLER.'

        utils.report_to_gh(
            title=gh_issue_title, 
            body=gh_issue_body
        )
    open_document(None, f'{utils.doc_flag_export_mark} at {utils.timestamp_now()}\n' + report_comment)


def serialize_to_csv():
    csv_lines = []
    for event_idx, rels in state.links.items():
        event = get_event_by_idx(event_idx)
        evt_id = event.id if event else f"{cur_doc.id}{GLUE}{utils.finalevt_export_label}"
        if utils.comment_ann_code in rels:
            comment, _ = rels[utils.comment_ann_code].copy().popitem()
            csv_lines.append(f'{evt_id},{utils.comment_ann_label},"{comment}"')
        if is_lightverb(rels):
            continue
        for rel, ents in rels.items():
            if rel != utils.comment_ann_code and rel > utils.notused_ann_code and ents:  # ignore all "unclaimed" or "not-food" entities after all
                if rel > 0:
                    relstr = str(rel)
                else:
                    relstr = utils.result_ann_label
                head_span, _ = ents.popitem()
                csv_lines.append(f'{head_span.id},{relstr},{evt_id}')
                for identity, whole in ents.items():
                    if identity.type == 'LOCATION':
                        relstr = utils.metonymy_export_label
                    elif not whole:
                        relstr = utils.meronymy_export_label
                    else:
                        relstr = utils.identity_export_label
                    csv_lines.append(f'{head_span.id},{relstr},{identity.id}')
                ents[head_span] = True
    return '\n'.join(csv_lines)


def open_document(new_docid, save_comment=None):
    if save_comment is not None:
        # means that the previous document is being closed, so save the final annotations
        # but before that, delete intermediate saves
        dataset.del_saved_annotation(state.cur_docid, state.username)
        dataset.save_serialization(state.cur_docid, state.username, save_comment, serialize_to_csv())
        dataset.save_graph(state.cur_docid, state.username, links_to_graph(True, False))
        dataset.save_dict_to_json(state.cur_docid, state.username, state.links)
    state.cur_docid = next_unannotated_docid() if new_docid is None else new_docid
    state.cur_evtnum = 0
    state.highlighted = None
    del state.observed
    del state.latest_status
    del state.links


def jump_back_to(event_idx):
    for i in range(state.cur_evtnum, event_idx - 1, -1):
        if i in state.observed:
            state.observed.pop(i)
        if i in state.links:
            state.links.pop(i)
    # when any popping happened in the above, finalizing event should also be popped
    if event_idx != state.cur_evtnum and -1 in state.links:
        state.links.pop(-1)
    state.latest_status = {}
    for observation in state.observed.values():
        state.latest_status.update(observation)
    state.cur_evtnum = event_idx
    state.highlighted = None


def annotation_sanity_check(links):
    # must have at least one "whole" (True) value
    errors = set()
    if not links.get(utils.result_ann_code) and not links.get(utils.participant_ann_code):
        errors.add(f' * Separation event must have a participant')
    for rel, ents in links.items():
        if isinstance(rel, int) and rel > utils.notused_ann_code:
            for ent in ents.copy().keys():
                if ent.type == 'EVENT' and ents[ent]:
                    ents.move_to_end(ent)
        if isinstance(rel, int) and rel > 0 and ents and not any(ents.values()):
            errors.add(f'* All of P.{rel} entities are only parts')
    return errors


def add_annotation(rel, argument, whole=True) -> Optional[str]:
    # returns error msg if any occured, otherwise returns None
    cur_links = state.links[state.cur_evtnum]
    if rel == utils.comment_ann_code:
        cur_links[rel] = odict({argument: whole})
    else:
        # remove from last radio btn click
        for ents in cur_links.values():
            if argument in ents.keys():
                ents.pop(argument)
        cur_links[rel][argument] = whole  # guaranteed to be placed at the end (prop of odict)
        if not whole or argument.type == 'LOCATION':
            cur_links[rel].move_to_end(argument, False)
    return None


def highlight_entity(ent):
    state.highlighted = ent


def links_to_graph(collapse_result=True, show_lights=True, grow_from_bottom=False):
    dot = graphviz.Digraph('graphp', graph_attr={'rankdir': 'BT' if grow_from_bottom else 'TB'})
    for evtI, rels in state.links.items():
        cur = evtI == state.cur_evtnum
        if is_separationevent(rels):
            add_multioutput_event_annotations_to_graph(dot, rels, current=cur, collapse_result=collapse_result)
        else:
            add_singleoutput_event_annotations_to_graph(dot, rels, collapse_result=collapse_result, show_lights=show_lights, current=cur, finalizing=evtI == -1)
    return dot


def infer_graphivz_edge_attrs(entity, whole_or_part):
    if entity.type == 'LOCATION':
        attrs = {
            'label': utils.METO_edge_label,
            'arrowhead': 'tee',
            'dir': 'back,'
        }
    elif entity.type == "HALT_CONDITION":
        attrs = {
            'label': utils.PROP_edge_label,
            'arrowhead': 'box',
            'dir': 'back,'
        }
    elif not whole_or_part:
        attrs = {
            'label': utils.MERO_edge_label,
            'arrowhead': 'icurve',
            'dir': 'back,'
        }
    else:
        attrs = {
            'label': utils.CUI_edge_label,
            'arrowhead': 'none'
        }
    return attrs


def add_multioutput_event_annotations_to_graph(graphdot:graphviz.Digraph, rels: Dict[int, OrderedDict[Union[str, Mention], bool]], collapse_result, current=False):
    evt_res, _ = rels[-1].popitem()  # this is guaranteed to exist as RES1.xxx, and always be "whole"
    main_anchor_str = None
    rels[-1][evt_res] = True
    if not collapse_result:
        main_anchor_str = span_to_str(evt_res, False)  # this is guaranteed to exist as RES1.xxx
    elif rels[utils.participant_ann_code]:
        main_input_span, _ = rels[utils.participant_ann_code].popitem()  # this is guaranteed to exist by our model of "one-to-many" separation
        main_anchor_str = span_to_str(main_input_span)
        rels[utils.participant_ann_code][main_input_span] = True
    elif utils.comment_ann_code in rels:  # when no input linked (light)
        main_anchor_str = span_to_str(evt_res, False)  # this is guaranteed to exist as RES1.xxx
    if main_anchor_str:
        graphdot.node(main_anchor_str, shape='oval' if collapse_result else 'rect', **(utils.cur_nodeattrs if current else {}))
        if utils.comment_ann_code in rels:
            evt_node_str = span_to_str(evt_res, affix_res=collapse_result) + (f'\n{utils.lightverb_export_label}' if is_lightverb(rels) else "")
            comment, _ = rels[utils.comment_ann_code].copy().popitem()
            comment_node_str = f'{comment} @ {evt_node_str}'
            graphdot.node(comment_node_str, **utils.comm_nodeattrs)
            graphdot.edge(f'{comment} @ {evt_node_str}', main_anchor_str, **utils.comm_edgeattrs)
    for rel, ents in rels.items():
        if not isinstance(rel, int):
            continue
        if (-100 < rel < 1) and ents:  # results
            head_span, _ = ents.popitem()
            dst_node_str = span_to_str(head_span)
            graphdot.node(dst_node_str, **(utils.hl_nodeattrs if state.highlighted == head_span else utils.cur_nodeattrs if current else {}))
            if not collapse_result:
                graphdot.edge(main_anchor_str, dst_node_str, f'{utils.phantom_result_prefix}{-rel}')
            elif main_anchor_str:
                graphdot.edge(main_anchor_str, dst_node_str, f'{evt_res.lemma}{utils.ann_delim}{utils.phantom_result_prefix}{-rel}')
            for ent_span, whole in ents.items():
                src_node_str = dst_node_str
                dst_node_str = span_to_str(ent_span)
                graphdot.node(dst_node_str, **(utils.hl_nodeattrs if state.highlighted == ent_span else {}))
                graphdot.edge(src_node_str, dst_node_str, **infer_graphivz_edge_attrs(ent_span, whole))
            ents[head_span] = True  # push the head of the coref-chain back in
        elif rel == utils.participant_ann_code and ents:
            head_span, _ = ents.popitem()
            src_node_str = span_to_str(head_span)
            graphdot.node(src_node_str, **(utils.hl_nodeattrs if state.highlighted == head_span else {}))
            if not collapse_result:
                graphdot.edge(src_node_str, main_anchor_str, utils.participant_ann_label)
            for ent_span, whole in ents.items():
                dst_node_str = span_to_str(ent_span)
                graphdot.node(dst_node_str, **(utils.hl_nodeattrs if state.highlighted == ent_span else {}))
                graphdot.edge(src_node_str, dst_node_str, **infer_graphivz_edge_attrs(ent_span, whole))
            ents[head_span] = True  # push the head of the coref-chain back in
        elif rel > 1:
            raise ValueError("Separation event can only be a one-to-many mapping!")


def add_singleoutput_event_annotations_to_graph(graphdot: graphviz.Digraph, rels: Dict[int, OrderedDict[Union[str, Mention], bool]], collapse_result, show_lights, finalizing=False, current=False):
    evt_res, _ = rels[utils.result_ann_code].popitem()  # this is guaranteed to exist by our model of "phantom" results
    evt_node_str = span_to_str(evt_res, affix_res=collapse_result) + (f'\n{utils.lightverb_export_label}' if is_lightverb(rels) else "")
    rels[utils.result_ann_code][evt_res] = True
    if show_lights or not is_lightverb(rels) or utils.comment_ann_code in rels:
        if collapse_result:
            graphdot.node(evt_node_str, **(utils.cur_nodeattrs if current else {}))
        elif finalizing:
            graphdot.node(utils.finalevt_export_label, shape='rect', **(utils.cur_nodeattrs if current else {}))
        else:
            graphdot.node(evt_node_str, shape='rect', **(utils.cur_nodeattrs if current else {}))
        if utils.comment_ann_code in rels:
            comment, _ = rels[utils.comment_ann_code].copy().popitem()
            comment_node_str = f'{comment} @ {evt_node_str}'
            graphdot.node(comment_node_str, **utils.comm_nodeattrs)
            graphdot.edge(f'{comment} @ {evt_node_str}', evt_node_str, **utils.comm_edgeattrs)
    for rel, ents in rels.items():
        if isinstance(rel, int) and rel > 0 and ents:  # positive int relations are "input"s
            head_span, _ = ents.popitem()
            src_node_str = span_to_str(head_span)
            graphdot.node(src_node_str, **(utils.hl_nodeattrs if state.highlighted == head_span else {}))
            edge_str = f'{utils.finalevt_export_label if finalizing else evt_res.text}{utils.ann_delim}{rel}' if collapse_result else str(rel)
            if finalizing:
                graphdot.edge(src_node_str, span_to_str(evt_res) if collapse_result else utils.finalevt_export_label, edge_str)
            else:
                graphdot.edge(src_node_str, evt_node_str, edge_str)
            for ent, whole in ents.items():
                graphdot.edge(src_node_str, span_to_str(ent),
                              **infer_graphivz_edge_attrs(ent, whole))
            ents[head_span] = True  # push the head of the coref-chain back in
        elif rel == utils.result_ann_code and not is_lightverb(rels):
            if not collapse_result:
                dst_node_str = span_to_str(evt_res)
                graphdot.node(dst_node_str, **(utils.hl_nodeattrs if state.highlighted == evt_res else {}))
                graphdot.edge(utils.finalevt_export_label if finalizing else evt_node_str, dst_node_str, utils.result_ann_label)
            for ent, whole in ents.items():
                # note that the "head" is at the end of this ordereddict, so need to ignore 
                if ent != evt_res:
                    dst_node_str = span_to_str(ent)
                    graphdot.node(dst_node_str, **(utils.hl_nodeattrs if state.highlighted == ent else {}))
                    graphdot.edge(evt_node_str, dst_node_str, **infer_graphivz_edge_attrs(ent, whole))


if len(state.documents) <= 0 or state.cur_docid is None:
    annotated_docs = dataset.list_annotated_docs(state.username)
    unassigned = dataset.list_unassigned_docs(state.username)
    st.write(f'You have annotated all {len(annotated_docs)} assigned documents ')
    st.write(f'There are {len(unassigned)} documents available for annotation.')
    st.write(f'Would you like to claim more documents?')
    st.button('Self-Assign more documents', on_click=dataset.assign_next_batch_to, args=[state.username])
    del state.cur_docid
    st.stop()


cur_doc = state.documents[state.cur_docid]
cur_event = get_cur_event()
final_result_name = dataset.get_document_title(cur_doc)

# `cur_entities` has two parts at the moment
# 1. carried over entities ("unclaimed") from previous steps of the current document
# 2. extent entities anchored in current step that are not yet linked to any event
# NOTE that all elements are extent entities, meaning they all must have positive `start` values
# during generating annotation table, the third part, "result" entities, will be generated 
cur_entities = [span for span in state.latest_status.keys()
                if isinstance(state.latest_status[span], int)
                and (utils.notused_ann_code <= state.latest_status[span] <= utils.result_ann_code)]
if cur_event:
    cur_entities += list(sorted(
        (span for span in cur_doc.entities
         if span.sent == cur_event.sent 
         and span not in state.latest_status.keys() 
         and span.start >= 0
         ),
        key=lambda x:x.start))


load_col, manual_col = st.columns((1,4))
with load_col:
    doc_to_annotate = list(state.documents.keys())
    idx = doc_to_annotate.index(cur_doc.id)
    st.write('Open Document')
    doc_id = st.selectbox('Open Document', state.documents.keys(), index=idx, label_visibility='collapsed')
    if doc_id != cur_doc.id:
        open_document(doc_id, save_comment=None)
        st.experimental_rerun()
if utils.guildelines:
    with manual_col:
        st.write('Annotation Instructions')
        is_url = urlparse(utils.guildelines)
        if is_url.scheme is not None and len(is_url.scheme) > 0 and is_url.scheme != 'file':
            st.write('Click [here]({}) to read the instructions'.format(utils.guildelines))
        else:
            with st.expander('Click here to read the instructions'):
                ann_inst_f = open(utils.guildelines)
                ann_inst = ann_inst_f.read()
                ann_inst_f.close()
                st.markdown(ann_inst)


text_col, table_col = st.columns((3, 2))
with text_col:
    st.subheader(final_result_name)
    st.write(cur_doc.url)
    color_coded_sents = []
    for sid, sent in enumerate(cur_doc.sentences, 1):
        # last one is always the result from the current event, and cur event will be highlighted with a different color
        ents_in_sent = [entity for entity in cur_entities if entity.sent == sid]  
        ents_in_sent.extend([e for e in cur_doc.events if e.sent == sid])
        # as `id` already contains sentId and start, using it as sorting key makes the list sorted by start, 
        # plus, non-`RES` entities come before `RES` entities. 
        ents_in_sent = sorted(ents_in_sent, key=lambda x: (x.sent, x.start))  
        colored_spans = []
        s = 0 
        seen = set()
        for ent in ents_in_sent:
            highlight_clicked = state.highlighted and ent.id in state.highlighted.id
            if ent.start not in seen:
                colored_spans.append(" ".join(t for t in sent[s:ent.start-1]))
                if ent.type == 'LOCATION':
                    k = 'location'
                elif ent.type == 'EVENT':
                    k = 'event'
                    if cur_event and ent.id == cur_event.id:
                        k = 'current'
                else:
                    k = 'entity'
                colored_spans.append((ent.text, f"{k}, {ent.start}", utils.annotated_text_colors['highlight'] if highlight_clicked else utils.annotated_text_colors[k]))
                seen.add(ent.start)
                s = ent.end - 1
            else:
                # conflicting "span" is guaranteed to be the last element of the highlights list 
                # because ents_in_sents is sorted by the start
                popped_text, popped_key, popped_color = colored_spans.pop()
                highlight_clicked = popped_color == utils.annotated_text_colors['highlight']
                # for `RES` entities, their id is suffixed with "res_exp_label + X" where X is multi-result numbering
                colored_spans.append((f'{popped_text} (RES{ent.id.rsplit(GLUE, 1)[1][len(utils.result_export_label):]})', popped_key, utils.annotated_text_colors['highlight'] if highlight_clicked else utils.annotated_text_colors['result']))
        colored_spans.append(" ".join(t for t in sent[s:]))
        color_coded_sent = f'{sid}. {annotated_text.util.get_annotated_html(*colored_spans)}'
        color_coded_sents.append(color_coded_sent)
    st.markdown('\n'.join(color_coded_sents), unsafe_allow_html=True)

    # "report" component
    if gh_enabled:
        st.warning("")
        with st.expander("**Does the document have a problem?**"):
            st.write("Use the input form below to mark this document as problematic.")
            issue_body = st.text_area("Comment on why this document is being flagged.", key=f'flagText.{cur_doc.id}')
            issue_title = f"{cur_doc.id} ({final_result_name}) is not annotatable"
            flag = st.button('REPORT', on_click=flag_and_next_document, args=[issue_body, issue_title],
                             disabled=not issue_body)

with table_col:
    # undo btn and event jump list 
    st.button('< Undo Last Event', on_click=jump_back_to, args=[state.cur_evtnum - 1 if state.cur_evtnum > 0 else len(cur_doc.events) - 1], disabled=state.cur_evtnum == 0)
    with st.expander(f'Current Event: **{cur_event.lemma if cur_event else utils.finalevt_export_label}** (click to see all events)', expanded=not debug and state.cur_evtnum == 0):
        if state.cur_evtnum == 0:
            st.write('This is the beginning of a document.')
            st.write('You can only jump to a past event (already annotated). '
                     'But taking a look at the full list of all events before starting'
                     'might be helpful to understand what to expect and what to annotate.')
        else:
            if state.cur_evtnum > 0:
                options = [(i, span_to_str(e, False)) for i, e in enumerate(cur_doc.events)][:state.cur_evtnum + 1]
                evt_jump_idx, _ = st.radio('jump back to (and undo annotations)', options=options, index=len(options) - 1)
            else:
                options = [(i, span_to_str(e, False)) for i, e in enumerate(cur_doc.events)]
                evt_jump_idx, _ = st.radio('jump back to (and undo annotations)', options=options)
            if 0 <= state.cur_evtnum != evt_jump_idx:
                jump_back_to(evt_jump_idx)
                st.experimental_rerun()
        st.radio('Future Events', [span_to_str(e, False) for e in cur_doc.events[state.cur_evtnum:]], disabled=True)

    # make the event a separation
    is_sep_evt = st.checkbox(
        'This is a separation event',
        key=f'separationEventCheck.{state.cur_evtnum}',
        help='Check this box if the event of interest is an event that splits or separates '
             'the participants into smaller pieces or sub-components.',
        on_change=jump_back_to,
        args=[state.cur_evtnum],
        value=False,
        disabled=state.cur_evtnum < 0,  # the final event cannot be separation
    )
    if is_sep_evt:
        if state.num_results == 1:
            state.num_results = 2
        result_export_suffixes = list(range(1, state.num_results + 1))
    else:
        state.num_results = 1
        result_export_suffixes = [""]
    if cur_event:
        res_ents = [Mention(f"{cur_event.id.rsplit(GLUE, 1)[0]}{GLUE}{utils.result_export_label}{suffix}", cur_event.sent, cur_event.start, cur_event.end, "EVENT", cur_event.lemma, cur_event.lemma)
                    for suffix in result_export_suffixes]
    else:
        res_ents = [Mention(f"{cur_doc.id}{GLUE}{utils.finalres_export_label}", None, -1, -1, "FINAL", final_result_name, final_result_name)]
    last_non_res_ent_idx = len(cur_entities) - 1
    cur_entities += res_ents
    
    # entity mark component
    r_pos = 0
    for i, entity in enumerate(cur_entities):
        base_label_options = (utils.notused_ann_code, utils.irrelevant_ann_code, utils.participant_ann_code if is_sep_evt else utils.result_ann_code)
        res_affix_num = 0  # do affix at all
        disabled = True
        if i <= last_non_res_ent_idx:
            if state.cur_evtnum < 0:
                default_option_idx = len(base_label_options) + r_pos
                r_pos += 1
            else:
                default_option_idx = 0
                disabled = False
        elif not is_sep_evt: 
            default_option_idx = base_label_options.index(utils.result_ann_code)
        else:
            default_option_idx = len(base_label_options) + r_pos
            r_pos += 1
        ent_label = span_to_str(entity)
        st.button(f'{ent_label} (click to highlight in text and graph)', on_click=highlight_entity, args=[entity], disabled=disabled, key=f'entHighlightBtn.{i}')
        # TODO (krim @ 1/6/2023): added "coreffereed by" line
        # [", ".join(ents[1:]) for rels in state.links.values() for rel, ents in rels.items() if -1000 < rel < 1  and ents[0] == entity]
        options = [utils.relation_labels[x] for x in base_label_options]
        extended_label_options_prefix = utils.output_ann_prefix if is_sep_evt else utils.input_ann_prefix
        if is_sep_evt:
            options += [f'{extended_label_options_prefix}{i+1}' for i in range(state.num_results)]
        else:
            options += [f'{extended_label_options_prefix}{i}' for i in range(1, len(cur_entities))]
        rel_label = st.radio(ent_label,
                             label_visibility='collapsed',
                             options=options,
                             key=f'linkRadio.{i}',
                             disabled=state.cur_evtnum < 0 or disabled,
                             index=default_option_idx,
                             horizontal=True,
                             )
        # make the entity as only "part"
        if not (disabled or entity.type == 'LOCATION'):
            is_whole = not st.checkbox(
                'Part-of',
                key=f'partOfCheck.{state.cur_evtnum}.{i}',
                help='Mark this entity as a "part-of" another entity in the same coreferent group',
                value=False,
            )
        else:
            is_whole = True
            if is_sep_evt and r_pos > 2:  # after 2nd result
                st.button('less splits', 
                          key=f'lessResults.{state.cur_evtnum}.{r_pos}',
                          on_click=decrease_numresults,
                          )
        if rel_label.startswith(utils.input_ann_prefix):
            code = int(rel_label.split(utils.ann_delim)[-1])
        elif rel_label.startswith(utils.output_ann_prefix):
            code = -int(rel_label.split(utils.ann_delim)[-1])
        else:
            code = utils.relation_codes[rel_label]
        add_annotation(code, entity, is_whole)
    if is_sep_evt:
        st.button('split more', 
                  key=f'moreResultsBtn',
                  on_click=increase_numresults,
                  )


    
    # free-text comment component
    # TODO (krim @ 12/20/22): see https://github.com/streamlit/streamlit/issues/678 for the "off-by-one" bug in streamlit
    # it's not trivial to use existing value for the default value of a newly generated `text_input` object 
    # because sometimes the new value is not properly propagated and defaults back to the old value
    # thus, I'm just going to empty the text box when the event step is reloaded. 
    new_comment = st.text_input('Any comment on this event?', key=f'commentText.{state.cur_evtnum}')
    if new_comment:
        add_annotation(utils.comment_ann_code, new_comment)
    st.warning('It takes split seconds to store comment in the memory. If you click the "next" button before the comment is stored, you\'ll see a "dictionary is empty" error. In such a case, please hit the "undo a step" and re-do the last step.')
    
    problems = annotation_sanity_check(state.links[state.cur_evtnum])
    # "done" component
    if state.cur_evtnum == -1:
        st.button('Save and Proceed', on_click=open_document, args=[None, None if debug else utils.timestamp_now()])
        if not debug:
            st.write('Save annotations and move on to a new next document (randomly picked from unannotated documents).')
    else:
        if is_lightverb(state.links[state.cur_evtnum]):
            st.button(f'Mark **{cur_event.lemma}** as a {utils.lightverb_export_label} event', on_click=next_step, disabled=len(problems) > 0)
        else:
            st.button(f'Done annotating **{cur_event.lemma}**', on_click=next_step, disabled=len(problems)>0)
    if problems:
        st.error('\n'.join(problems))
    

st.header("Current Transformation Graph")

# visualization preferences
state.viz_pref_nodify_events = st.checkbox(
    'Explicit Event Nodes',
    key='nodifyEventsCheck', 
    # TODO (krim @ 1/2/23): this, and following viz_options were all sufferring 
    # from the streamlit "off-by-one" issue, so I'm disabling setting the default value
    # value=state.get('viz_pref_nodify_events', False),
    help='When checked, events are represented as independent nodes. Otherwise, events are represented only as directed edges from inputs to an output.',
)
state.viz_pref_show_lights = st.checkbox(
    f'Show {utils.lightverb_export_label} Event Nodes ',
    key='showLightsCheck',
    # value=state.get('viz_pref_show_lights', True),
    help=f'A {utils.lightverb_export_label} Event is an event that takes no "inputs". See the guidelines above for more details.'
)
state.viz_pref_bt_growth = st.checkbox(
    f'Graph grows from bottom to top',
    key='btGrowthCheck',
    # value=state.get('viz_pref_show_lights', True),
    help=f'Be default, graphs are rendered from top to bottom. Checking here makes the growth inverse.'
)

st.graphviz_chart(
    links_to_graph(
        collapse_result=not state.viz_pref_nodify_events,
        show_lights=state.viz_pref_show_lights,
        grow_from_bottom=state.viz_pref_bt_growth,
    )
)
if debug:
    for line in serialize_to_csv().split('\n'):
        st.write(line)
