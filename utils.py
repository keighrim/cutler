import datetime
import json
import urllib.parse
import urllib.request
from pathlib import Path

import yaml


annotated_text_colors = {'entity': '#7799ff',
                         'location': '#aaccff',
                         'current': '#ff6699',
                         'event': '#eeeeee',
                         'result': '#bbaaee',
                         'highlight': '#00ffaa'}
# some graphviz attribs
hl_nodeattrs = {'color': annotated_text_colors['highlight'], 'penwidth': '4.0'}
cur_nodeattrs = {'color': annotated_text_colors['current'], 'penwidth': '4.0'}
comm_nodeattrs = {'shape': 'plaintext'}
comm_edgeattrs = {'arrowhead': 'none', 'style': 'dashed', 'color': 'grey'}

phantom_result_prefix = 'RES'
# constants for reserved relation labels
notused_ann_label = 'USE LATER'
notused_ann_code = -1000
irrelevant_ann_label = 'DISCARD'
irrelevant_ann_code = -1001
comment_ann_label = 'COMMENT'
comment_ann_code = -2000
result_ann_label = 'RESULT'
result_ann_code = 0
participant_ann_label = 'PARTICIPANT'
participant_ann_code = 1
ann_delim = '.'
output_ann_prefix = f'R{ann_delim}'
input_ann_prefix = f'P{ann_delim}'
# with separation event, 0 is discarded and negative ints (from -1) will be used

relation_labels = {
    comment_ann_code: comment_ann_label,
    notused_ann_code: notused_ann_label,
    irrelevant_ann_code: irrelevant_ann_label,
    result_ann_code: result_ann_label,
    participant_ann_code: participant_ann_label,
}

relation_codes = {v: k for k, v in relation_labels.items()}

finalevt_export_label = '@FINALIZE'
finalres_export_label = 'FINALRES'
result_export_label = 'CUTRES'
identity_export_label = 'CUI'
meronymy_export_label = 'MERONYM'
metonymy_export_label = 'METONYM'
propery_export_label = 'PROP'
lightverb_export_label = 'LIGHT'
doc_flag_export_mark = 'FLAGGED'
CUI_edge_label = 'identity'
METO_edge_label = 'metonym'
MERO_edge_label = 'part-of'
PROP_edge_label = 'property'


def load_conf(conf_fname):
    conf_f = open(conf_fname)
    conf = yaml.load(conf_f, Loader=yaml.SafeLoader)
    conf_f.close()
    return conf


def save_conf(conf_fname, conf):
    conf_f = open(conf_fname, 'w')
    yaml.dump(conf, conf_f, default_flow_style=False)
    conf_f.close()


cutler_conf_fname = Path(__file__).parent / 'config.yaml'
cutler_conf = load_conf(cutler_conf_fname)  # will contain app configuration such github login token for flagging
guildelines = cutler_conf['guidelines']

token = cutler_conf['ghtoken']
org = cutler_conf['ghorg']
repo = cutler_conf['ghrepo']
issue_labels = cutler_conf['ghlabels']


def get_existing_gh_issues():
    url = f'https://api.github.com/repos/{org}/{repo}/issues?state=all'
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    req = urllib.request.Request(url, headers=headers)
    res = urllib.request.urlopen(req)
    res_json = json.load(res)
    existing_issues = {issue['title']: (issue['number'], issue['state'] == 'open') for issue in res_json}
    return existing_issues
    
    
def add_comment_to_gh_issue(issue_num, body, reopen=True):
    url = f'https://api.github.com/repos/{org}/{repo}/issues/{issue_num}/comments'
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    issue = {
        'body': body,
    }
    req = urllib.request.Request(url, data=json.dumps(issue).encode('utf8'), headers=headers, method='POST')
    
    res = urllib.request.urlopen(req)

    if reopen:
        url = f'https://api.github.com/repos/{org}/{repo}/issues/{issue_num}'
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"token {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        issue = {
            'state': 'open',
        }
        req = urllib.request.Request(url, data=json.dumps(issue).encode('utf8'), headers=headers, method='PATCH')
        res = urllib.request.urlopen(req)


def create_gh_issue(title, body):
    url = f'https://api.github.com/repos/{org}/{repo}/issues'
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    issue = {
        'title': title,
        'body': body,
        'labels': issue_labels,
    }
    req = urllib.request.Request(url, data=json.dumps(issue).encode('utf8'), headers=headers, method='POST')
    res = urllib.request.urlopen(req)
    
    
def report_to_gh(title, body):
    existings = get_existing_gh_issues()
    if title in existings:
        add_comment_to_gh_issue(existings[title][0], body)
    else:
        create_gh_issue(title, body)


def timestamp_now():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')

