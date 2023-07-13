import datetime
import itertools
import pathlib
import re
import tempfile
import urllib.error
import zipfile
from collections import defaultdict as ddict

import graphviz
import pandas
import streamlit as st
from PIL import Image

import dataset
import utils

admins = {}
st.set_page_config(layout="wide", page_icon='🍽️', page_title=f'CUTLER Dashboard', initial_sidebar_state="collapsed")

dataset_conf, assignments_per_annotator = dataset.load_dataset_conf()

if 'graph_shown' not in st.session_state:
    st.session_state['graph_shown'] = None


def set_graph_shown(doc_id):
    st.session_state['graph_shown'] = doc_id
    

if not st.session_state.authentication_status:
    st.subheader("User not found")
    st.write("It seems that we can't find your log-in information."
             "Did you log in via the main app page? ")
    st.stop()


def zip_all(filelist, zipdir, zipname=None):
    zipfname = 'annotations.zip' if zipname is None else zipname
    if not zipfname.endswith('.zip'):
        zipfname += '.zip'
    zipfname = pathlib.Path(zipdir)/zipfname
    zipf = zipfile.ZipFile(zipfname, 'w', zipfile.ZIP_DEFLATED)
    for f in filelist:
        zipf.write(f, f.name)
    zipf.close()
    return zipfname


def get_files_for_download(annotator_name: str = None, include_png: bool = True,):
    # TODO (krim @ 1/7/23): add a time-based filter? `after: datetime.datetime = datetime.datetime(1970, 1, 1)`
    glob_pat = "*" if annotator_name is None else f"*.{annotator_name}*"
    exts = ['csv']
    if include_png:
        exts.append('png')
    return itertools.chain(*(dataset.CUTLER_ANNOTATION_PATH.glob(f'{glob_pat}.{ext}') for ext in exts))


try:
    existing = utils.get_existing_gh_issues()
except urllib.error.HTTPError:
    existing = {}
    
    
stats_col, dl_col = st.columns([9, 1])
with dl_col:
    dl_all = st.button('Download ALL Annotations')
    if dl_all:
        with tempfile.TemporaryDirectory() as tempd:
            zipfname = zip_all(get_files_for_download(annotator_name=None), tempd, f'annotations-all-{utils.timestamp_now()}')
            with open(zipfname, 'rb') as dl_f:
                st.download_button('**ALL Annotations** are ready to download', data=dl_f, file_name=zipfname.name)
            
    dl_you = st.button('Download YOUR Annotations')
    if dl_you:
        u = st.session_state['username']
        with tempfile.TemporaryDirectory() as tempd:
            zipfname = zip_all(get_files_for_download(annotator_name=u), tempd, f'annotations-{u}{utils.timestamp_now()}')
            with open(zipfname, 'rb') as dl_f:
                st.download_button(f'**{u} Annotations** are ready to download', data=dl_f, file_name=zipfname.name)

total = 0
fully_assigned = 0
adjudicated = 0
fully_done = []
fully_flagged = []
partially_flagged = []
flagged_and_finished = []
flagged_per_annotator = ddict(int)
finished_per_annotator = ddict(int)
for doc_id, annotators in dataset_conf['assignments'].items():
    if len(annotators) < 1:
        continue
    if doc_id in dataset.ignored_docs:
        continue
    total += 1
    fullfillreq = dataset_conf['annotatorsPerDocument']
    if len(annotators) == fullfillreq:
        fully_assigned += 1
    flags = 0
    dones = 0
    doc_title = dataset.get_document_title(dataset.docs[doc_id])
    title_cont = st.container()
    gold_exist = False
    if dataset.get_save_fname(doc_id, 'gold').exists():
        adjudicated += 1
        gold_exist= True
    annotators.sort()
    cols = st.columns(fullfillreq)
    for i, col in enumerate(cols):
        with col:
            if i == len(annotators):
                st.button(f'ASSIGN {doc_id} to me', 
                          key=f'selfAssignBtn-{doc_id}',
                          on_click=dataset.assign_single_document_to,
                          args=[doc_id, st.session_state['username']],
                          disabled=st.session_state['username'] in annotators,
                          )
            else:
                ann = annotators[i]
                status = 'undone'
                timestamp = ""
                comment = []
                if dataset.is_annotated(doc_id, ann):
                    save_fname = dataset.get_save_fname(doc_id, ann)
                    with open(save_fname) as save_f:
                        for line in save_f:
                            if line.startswith('#'):
                                comment.append(line[1:].strip())
                            else:
                                break
                    if comment and utils.doc_flag_export_mark in comment[0]:
                        status = 'FLAGGED'
                        flags += 1
                        flagged_per_annotator[ann] += 1
                    else:
                        status = 'FINISHED'
                        dones += 1
                        finished_per_annotator[ann] += 1
                    m = re.search(r'\d{6}-\d{6}', comment[0])
                    if m:
                        timestamp = datetime.datetime.strptime(m[0], '%y%m%d-%H%M%S').strftime("%m/%d/%Y %H:%M")
                        
                iss_num = None
                iss_open = True
                if status == 'FLAGGED':
                    issue_title = f"{doc_id} ({doc_title}) is not annotatable"
                    if issue_title in existing:
                        iss_num, iss_open = existing[issue_title]
                with st.expander(f'{ann} ({status}{" at " if timestamp else ""}{timestamp}{"" if iss_open else " (issue resolved)"})', expanded=st.session_state['graph_shown'] == doc_id):
                    if comment:
                        for line in comment:
                            st.write(line)
                    if status == 'undone':
                        unassign = st.button(f"Unassign {doc_id} from {ann}",
                                           key=f'unassignBtn-{doc_id}-a{i}',
                                           on_click=dataset.unassign_document_from,
                                           args=[doc_id, ann],
                                           help='will unassign this document from the current anntoator',
                                           )
                    else:
                        delete = st.button(f"DELETE {doc_id}", 
                                           key=f'deleteBtn-{doc_id}-a{i}',
                                           on_click=dataset.del_saved_annotation, 
                                           args=[doc_id, st.session_state['username']],
                                           disabled=ann not in admins and ann != st.session_state['username'],
                                           help='will delete the annotation WITHOUT asking for confirmation!',
                                           )
                    if status == 'FLAGGED':
                        if iss_num:
                            st.markdown(f'[See the discussion on Github](https://github.com/brandeis-llc/coref-under-transformation/issues/{iss_num})')
                            if not iss_open:
                                st.markdown(f'#### :red[NOTE As the issue is already closed, the reported issue is probably now fixed. ]')
                    st.button(f'Show Annotation',
                              key=f'showBtn-{doc_id}-a{i}',
                              disabled=status == 'undone',
                              on_click=set_graph_shown, 
                              args=[doc_id],
                              )
                    if st.session_state['graph_shown'] == doc_id and dataset.is_annotated(doc_id, ann):
                        st.markdown(dataset.get_commented_document_text(doc_id, ann))
                        png_fname = dataset.get_save_fname(doc_id, ann, 'gvz.png')
                        gvz_fname = dataset.get_save_fname(doc_id, ann, 'gz')
                        if png_fname.exists():
                            i = Image.open(png_fname)
                            st.image(i)
                        elif gvz_fname.exists():
                            st.graphviz_chart(graphviz.Source.from_file(gvz_fname).source)
                        else:
                            st.error('Graphviz source or image file is not found. Probably something went wrong during saving annotations. '
                                     'Cannot display a graph.')

    if dones == fullfillreq:
        fully_done.append(doc_id)
        color = 'blue'
    if flags == fullfillreq:
        fully_flagged.append(doc_id)
        color = 'red'
    elif flags > 0:
        color = 'orange'
        if dones > 0:
            flagged_and_finished.append(doc_id)
        else:
            partially_flagged.append(doc_id)
    elif gold_exist:
        color = 'green'
    else: 
        color = 'black'
    title_cont.write(f':{color}[{doc_id}: {doc_title}]')
        
with stats_col:
    with st.expander("See dataset statistics"):
        st.write(f'TOTAL: {total}')
        st.write(f'Fully assigned: {fully_assigned}')
        st.write(f'All annotators finished: {len(fully_done)}')
        st.caption(fully_done)
        st.write(f'Adjudicated: {adjudicated}')
        st.write(f'All annotators flagged: {len(fully_flagged)}')
        st.caption(fully_flagged)
        st.write(f'Some annotators flagged: {len(partially_flagged)}')
        st.caption(partially_flagged)
        st.write(f'Some annotators flagged, some finished: {len(flagged_and_finished)}')
        st.caption(flagged_and_finished)
        st.write('Document removed from document for un-resolvable issues')
        st.write(dataset.ignored_docs)
        df = []
        total_flags = 0
        total_dones = 0
        _, ass_per_annotator = dataset.load_dataset_conf()
        for ann in set().union(flagged_per_annotator.keys(), finished_per_annotator.keys()):
            df.append(dict(zip(('name flags dones assigned'.split()), (ann, flagged_per_annotator[ann], finished_per_annotator[ann], len(ass_per_annotator[ann] - dataset.ignored_docs)))))
            total_flags += flagged_per_annotator[ann]
            total_dones += finished_per_annotator[ann]
        df.append(dict(zip(('name flags dones assigned'.split()), ('total', total_flags, total_dones, 0))))
        st.dataframe(pandas.DataFrame(df))
        st.write("Title colors below: red - all flagged, orange - some flagged, blue - all finished, green - adjudicated")
