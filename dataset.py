import json
import math
import os
import random
from collections import defaultdict as ddict
from collections import namedtuple as nt
from pathlib import Path

import graphviz

import utils

GLUE = '.'
# a list of doc_ids that should be ignored for January release of the CUTL dataset
ignored_docs = {}

Mention = nt('Mention', ['id', 'sent', 'start', 'end', 'type', 'text', 'lemma'])
Document = nt('Document', ['id', 'url', 'entities', 'events', 'sentences'])

dataset_conf_fname = Path(__file__).parent / 'dataset.yaml'


def load_dataset_conf():
    dataset_conf = utils.load_conf(dataset_conf_fname)

    assignments_per_annotator = ddict(set)
    for doc_id, a_names in dataset_conf['assignments'].items():
        for a_name in a_names:
            assignments_per_annotator[a_name].add(doc_id)
    return dataset_conf, assignments_per_annotator


def save_dataset_conf():
    global conf
    utils.save_conf(dataset_conf_fname, conf)


conf, assignments_per_annotator = load_dataset_conf()
CUTLER_ANNOTATION_PATH = Path(conf['datasetDirectory']) / 'output'


def load_document(doc_fname):
    with open(doc_fname, 'r') as f:
        j = json.load(f)
        did = Path(doc_fname).with_suffix("").name
        url = j['sourceUrl']
        entities = []
        events = []
        sentences = [sent['tokens'] for sent in j['sentences']]
        for mention in j['mentions']:
            mid = GLUE.join([mention['location'], mention['type']])
            sent, start, length = list(map(int, mention['location'].split(GLUE)))
            end = start + length
            lemma = " ".join(j['sentences'][sent-1]['lemmas'][start-1:end-1])
            m = Mention(mid, sent, start, end, mention['type'], mention['text'], lemma)
            if mention['type'].upper() == 'EVENT':
                events.append(m)
            else:
                entities.append(m)
            # sort mention lists by the order of their appearance in the document
            for mentions in (events, entities):
                mentions.sort(key=lambda x: (x.sent, x.start))
        return Document(did, url, entities, events, sentences)


docs = {}
for doc_fname in Path(conf['datasetDirectory']).glob("*.json"):
    doc = load_document(doc_fname)
    # if doc.id not in ignored_docs:
    docs[doc.id] = doc


def add_annotator(annotator_name):
    global conf
    conf, assignments_per_annotator = load_dataset_conf()
    if annotator_name not in conf['annotators']:
        assign_next_batch_to(annotator_name)


def unassign_document_from(doc_id, annotator_name):
    global conf
    conf, assignments_per_annotator = load_dataset_conf()
    cur_assigned = conf['assignments'].get(doc_id, [])
    if annotator_name in cur_assigned and not get_save_fname(doc_id, annotator_name).exists():
        cur_assigned.remove(annotator_name)
        conf['assignments'][doc_id] = cur_assigned
        assignments_per_annotator[annotator_name].remove(doc_id)
    save_dataset_conf()


def assign_document_to(doc_id, annotator_name):
    global conf
    cur_assigned = conf['assignments'].get(doc_id, [])
    if len(cur_assigned) < conf['annotatorsPerDocument'] and annotator_name not in cur_assigned:
        cur_assigned.append(annotator_name)
        conf['assignments'][doc_id] = cur_assigned
        assignments_per_annotator[annotator_name].add(doc_id)
    save_dataset_conf()


def assign_batch_to(doc_ids, annotator_name):
    for doc_id in doc_ids:
        assign_document_to(doc_id, annotator_name)


def assign_single_document_to(doc_id, annotator_name):
    assign_document_to(doc_id, annotator_name)


def assign_next_batch_to(annotator_name, batch_size=conf['batchSize'], randompick=True):
    global conf, assignments_per_annotator
    # TODO (krim @ 12/29/2022): currently there's no handling of concurrent YAML file access
    conf, assignments_per_annotator = load_dataset_conf()
    if annotator_name not in conf['annotators']:
        conf['annotators'].append(annotator_name)
    assign_batch_to(list_unassigned_docs(annotator_name, batch_size, randompick), annotator_name)
    save_dataset_conf()


def list_unassigned_docs(annotator_name=None, batch_size=math.inf, randompick=False):
    global conf
    conf, assignments_per_annotator = load_dataset_conf()
    if not annotator_name:
        # TODO (krim): also exclude docs in ignored list 
        return list(docs.keys() - conf['assignments'].keys())
    else:
        # TODO (krim): also exclude docs in ignored list 
        assigned_but_notannotated = sum(
            not is_annotated(doc_id, annotator_name) 
            for doc_id in assignments_per_annotator[annotator_name])
        batch_size = max(0, batch_size - assigned_but_notannotated)
        unassigned = []
        for doc_id in docs.keys():
            if len(unassigned) >= batch_size:
                break
            # if doc_id not in ignored_docs:
            cur_assigned = conf['assignments'].get(doc_id, [])
            if len(cur_assigned) < conf['annotatorsPerDocument'] and annotator_name not in cur_assigned:
                unassigned.append(doc_id)
        if randompick: 
            random.shuffle(unassigned)
        return unassigned


def list_unannotated_docs(annotator_name):
    unannotated = [doc_id for doc_id in assignments_per_annotator[annotator_name] if not is_annotated(doc_id, annotator_name)]
    return unannotated


def list_annotated_docs(annotator_name):
    return [path.name.rsplit('.', 2)[0] for path in CUTLER_ANNOTATION_PATH.glob(f"*.{annotator_name}.csv")]


def read_unannotated_documents(annotator_name):
    return {doc_id: docs[doc_id] for doc_id in list_unannotated_docs(annotator_name) if doc_id not in ignored_docs}


def get_document_title(document):
    return document.url.split('/')[-1]


def get_save_fname(doc_id, annotator_name, ext='csv'):
    if ext == '*':
        return CUTLER_ANNOTATION_PATH.glob(f"{doc_id}.{annotator_name}.*")
    return CUTLER_ANNOTATION_PATH / f"{doc_id}.{annotator_name}.{ext}"


def save_dict_to_json(doc_id, annotator_name, link_dict):
    new_dict = {}
    for evtid, rels in link_dict.items():
        if evtid not in new_dict:
            new_dict[evtid] = {}
        for rel, spans in rels.items():
            new_dict[evtid][rel] = list(map(lambda x: (x.id if isinstance(x, Mention) and x.type.upper() != "EVENT" else x, spans[x]), spans))
    with open(get_save_fname(doc_id, annotator_name, 'json'), 'w') as out_f:
        json.dump(new_dict, out_f, indent=2)
            

def save_serialization(doc_id, annotator_name, save_comment, serialized_annotation):
    CUTLER_ANNOTATION_PATH.mkdir(exist_ok=True, parents=True)
    with open(get_save_fname(doc_id, annotator_name), 'w') as out_f:
        for line in save_comment.split('\n'):
            out_f.write(f'# {line}\n')
        for line in serialized_annotation.split('\n'):
            out_f.write(f'{line}\n')


def save_graph(doc_id, annotator_name, dot_obj):
    dot_obj.format = 'png'
    try:
        dot_obj.render(filename=get_save_fname(doc_id, annotator_name, 'gvz'))
    except graphviz.ExecutableNotFound:
        pass


def del_saved_annotation(doc_id, annotator_name):
    list(map(os.remove, get_save_fname(doc_id, annotator_name, "*")))


def is_annotated(doc_id, annotator_name):
    return get_save_fname(doc_id, annotator_name).exists()


def get_commented_document_text(doc_id, annotator_name=None):
    comments = ddict(list)
    if annotator_name:
        with open(get_save_fname(doc_id, annotator_name)) as save_f:
            for line in save_f:
                if utils.comment_ann_label in line:
                    evt_id, _, comment = line.split(',', maxsplit=2)
                    sent_id = evt_id.rsplit(GLUE, maxsplit=1)[0]
                    comments[sent_id].append(comment)
    sent_strs = []
    for i, sent in enumerate(docs[doc_id].sentences):
        sent_strs.append(f"{i}. {sent}")
        # TODO (krim): fix this to get cur_sent number without using old send class
        for comment in comments[sent.id]:
            sent_strs.append(f"    * {comment}")
    return "\n".join(sent_strs)
