# CULTER - Coreference Under Transformatino Labeler

CUTLER is an annotation environment that was designed, developed in used in CUTL annotation (TODO, cite paper)

## Citation and License


This software is developed as an annotation environment for the following study: 

[The Coreference under Transformation Labeling Dataset: Entity Tracking in Procedural Texts Using Event Models](https://aclanthology.org/2023.findings-acl.788) (Rim et al., Findings 2023) [[bib](https://aclanthology.org/2023.findings-acl.788.bib)]

and distributed under the [GPLv3](LICENSE) license. Please cite the above paper if you use this software in your research.

## Manuals

This is a manual for annotation project managers who want to use the CUTLER annotation environment. For users (annotators), at the moment the manual for user are combined with the CUTL annotation guidelines. See [CUTL annotation guidelines](https://github.com/brandeis-llc/dp-cutl/blob/main/guidelines.md). 

### Requiremnts and Installtiona

* CUTLER is developed on Python 3.10, but should work on any Python 3.8+ environment.
* At the moment, all annotation data I/O is done via local file system, hence no database software is required. 
* Annotators will use web interface, meaning a manager needs a server to host the annotation environment.
* The web interface is based on [Streamlit](https://streamlit.io/).
* Python dependencies are listed in `requirements.txt` file. Install them with `pip install -r requirements.txt` command.

### Server configuration

To configure the server environment, one should use [`.streamlit/config.toml`](https://docs.streamlit.io/library/advanced-features/configuration). CUTLER will work with the default configuration, but you may want to change `server.port`, `server.baseUrlPath` and `server.enableCORS` options depending on your server security policy. One thing we recommend is to disable *dark* theme by setting `theme.base="light"`, as the color scheme of the annotation interface is not optimized for dark theme.

### Data preparation

Input data files must be in the following format:

1. file must be in JSON format
2. file name should be <DOC_ID>.json
3. at the top level; 
    1. `sentences`: list of sentence objects
    2. `mentions`: list of mention object
    3. (optional) `sourceUrl`: the document source URL (e.g. the URL of the web page where the document is from)
4. sentence object
    1. `tokens`: list of tokens, a token is whitespace separated text
    2. `lemmas`: list of lemmas, must be 1-on-1 mappable with tokens
    3. `pos`: list of part-of-speech tags, must be 1-on-1 mappable with tokens
5. mention object
   1. `location`: dot-joined string of 1. sentence index, 2. token index, 3. mention length. Add indices are 1-based. e.g. `"1.2.3"` means 3 tokens long mention starting from 2nd token of 1st sentence.
   2. `type`: mention label, must be one of `EVENT`, `ENTITY`, `LOCATION`
6. both sentence object and mention object can have any number of additional fields, which will be simply ignored by the CUTLER, hence will be lost in the output file.

Here's an example of what an input file looks like: 

``` json 
{
  "sentences": [
    {
      "text": "Wash the chickpeas, drain, cover with water and let soak overnight.",
      "tokens": [
        "Wash", "the", "chickpeas", ",", "drain", ",", "cover", "with", "water", "and", "let", "soak", "overnight", "."
      ],
      "lemmas": [
        "wash", "the", "chickpea", ",", "drain", ",", "cover", "with", "water", "and", "let", "soak", "overnight", "."
      ],
      "pos": [
        "VERB", "DET", "NOUN", "PUNCT", "VERB", "PUNCT", "VERB", "ADP", "NOUN", "CCONJ", "VERB", "VERB", "ADV", "PUNCT"
      ]
    },
    # more sentences
  ],
  "mentions": [
    { "location": "1.1.1", "type": "EVENT", "text": "Wash" },
    { "location": "1.3.1", "type": "ENTITY", "text": "chickpeas" },
    { "location": "1.5.1", "type": "EVENT", "text": "drain" },
    { "location": "1.7.1", "type": "EVENT", "text": "cover" },
    { "location": "1.9.1", "type": "ENTITY", "text": "water" },
    { "location": "1.12.1", "type": "EVENT", "text": "soak" },
    # more mentions
  ]
}
```

### Application Configuration

#### Annotator management: `annotators.yaml`

At the moment, all annotators are managed via `annotators.yaml` file. The GUI does NOT provide any interface for annotator creation, so all new annotators must be manually entered to this file with some default password. Once the annotator is created, the annotator can change their password via the GUI. See the included `annotators.yaml` file for the format to add annotators.

#### Dataset management: `dataset.yaml`

Dataset I/O is managed via `dataset.yaml` file. The file contains these keys; 

* `annotatorsPerDocument`: number of annotators to assign to each document, to support duplicate annotation. At the moment, CUTLER does not provide inter-annotator agreement calculation nor adjudication interface, so this is just a number to support duplicate annotation.
* `batchSize`: number of documents to assign to each annotator at once. Annotators can self-assing more documents once they finish their assigned documents.
* `datasetDirectory`: path to the dataset directory. This directory must contain all json files prepared for the annotation. When using a relative path, it is relative to the working directory where `streamlit run` command is executed.
* `annotators`: managed by CUTLER, leave as `[]` at the first time.
* `assignments`: managed by CUTLER, leave as `{}` at the first time.

#### Misceillaneous configurations: `config.yaml`

##### On-screen annotation guidelines 

Use `guidelines` key to provide on-screen annotation guidelines. The value must be a single string, either a URL to an online resource, or a local file name. If a local file name is given, CUTLER will try to render the file as markdown. Annotators will see a button to quickly open the guidelines in the annotation interface.

##### Issue reporting

CUTLER uses GitHub issues to manage annotation issues. The `config.yaml` file contains configuration keys for the GitHub repository information and the issue labels to use (`ghxxx` keys). If you don't want to use GitHub issues, you can leave the file as is, and CUTLER will disable the issue reporting button. 

### Data output format

CUTLER will write output files in `<dataset>/output` directory (`<dataset>` is the configured directory). The output files are named after `<doc_id>.<annotattor_username>.<extension>`. 

(For intermediate save files, `intermediate.` is appended to the extension name)

For each extension, the output file will contain the following data:

* `gvz`: graphviz dot file for the annotated graph
* `png`: png rendering of the above dot file
* `json`: raw "dump" of internal data structure, only useful for debugging and recovering annotation sessions (not implemented yet)
* `cvs`: trimmed output of annotation. Each row is a triple of an edge relation from the annotated graph. 
    * The relations (the middle column) used in the export format are 
        * natural numbers: A is a participant of event B
        * `RESULT`: A is a result of event B
        * `CUI`: A is identical of B
        * `METONYM`: A is a metonym of B
        * `MERONYM`: A is a meronym of B
        * `PROP`: A is a property of B
    * The both ends of relations (participants) are formatted as `<locaion>.<type_suffix>` of the mention.
    * Suffixes used in both ends of relations (the first and last column)
        * `OBEJCT`: the node is an entity from a text extent
        * `EVENT`: the node is an event
        * `CUTRES`: the node is a result entity, derived from an event
        * `@FINALIZE` and `FINALRES`: these are special labels used for the final nodes in the document graph. The final `RES.` entity will always participate a dummy `@FINALIZE` event that results in another dummy `FINALRES` node, and this row should conclude the annotation for the document.
