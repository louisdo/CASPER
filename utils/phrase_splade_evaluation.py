# We reuse some of the code structures of SummEval (https://github.com/Yale-LILY/SummEval)

import os
import json
import numpy as np
import argparse
import gin
import sys
import os
sys.path.append('/home/lvnguyen/research/Efficient-Storage-for-Large-IR-Systems') # Change to your Efficient-Storage-for-Large-IR-Systems dir

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def cli_main():
    parser = argparse.ArgumentParser(description="predictor")
    parser.add_argument('--config-file', type=str, help='config file with metric parameters')
    parser.add_argument('--metrics', type=str, help='comma-separated string of metrics')
    parser.add_argument('--input-file', type=str, help='file containing the input document, one document per line')
    parser.add_argument('--label-file', type=str, help='file containing the reference keyphrases, separated by semicolon (;)')
    parser.add_argument('--output-file', type=str, help='file containing the model\'s predicted keyphrases, separated by semicolon (;)')
    parser.add_argument('--jsonl-file', type=str, help='input jsonl file to score', default=None)
    parser.add_argument('--log-file-prefix', type=str, help='output score log file')
    parser.add_argument('--only-use-first-n', type=int, default=None, help='only use the first n example for evaluation')
    args = parser.parse_args()

    gin.parse_config_file(args.config_file)
    
    metrics = [x.strip() for x in args.metrics.split(",")]
    metrics_dict = {}
    
    if "approximate_matching" in metrics:
        from metrics import ApproximateMatchingMetric
        metrics_dict["approximate_matching"] = ApproximateMatchingMetric()
        
    if "bert_score" in metrics:
        from metrics import BertScoreMetric
        metrics_dict["bert_score"] = BertScoreMetric()
        
    if "chatgpt" in metrics:
        from metrics import ChatGPTMetric
        metrics_dict["chatgpt"] = ChatGPTMetric()

    if "diversity" in metrics:
        from metrics import DiversityMetric
        metrics_dict["diversity"] = DiversityMetric()
        
    if "exact_matching" in metrics:
        from metrics import ExactMatchingMetric
        metrics_dict["exact_matching"] = ExactMatchingMetric()
    
    if "fg" in metrics:
        from metrics import FGMetric
        metrics_dict["fg"] = FGMetric()

    if "meteor" in metrics:
        from metrics import MeteorMetric
        metrics_dict["meteor"] = MeteorMetric()
        
    if "mover_score" in metrics:
        from metrics import MoverScoreMetric
        metrics_dict["mover_score"] = MoverScoreMetric()
        
    if "retrieval" in metrics:
        from metrics import RetrievalMetric
        # hack here, remove later
        if (args.output_file and 'exhird' in args.output_file) or (args.jsonl_file and 'exhird' in args.jsonl_file):
            metrics_dict["retrieval"] = RetrievalMetric(force_recalc_index=True)
        elif (args.output_file and 'hypermatch' in args.output_file) or (args.jsonl_file and 'hypermatch' in args.jsonl_file):
            metrics_dict["retrieval"] = RetrievalMetric(force_recalc_index=True)
        elif (args.output_file and 'segnet' in args.output_file) or (args.jsonl_file and 'segnet' in args.jsonl_file):
            metrics_dict["retrieval"] = RetrievalMetric(force_recalc_index=True)
        else:
            metrics_dict["retrieval"] = RetrievalMetric()
        
    if "retrieval_v2" in metrics:
        from metrics import RetrievalMetricV2
        # hack here, remove later
        if (args.output_file and 'exhird' in args.output_file) or (args.jsonl_file and 'exhird' in args.jsonl_file):
            metrics_dict["retrieval_v2"] = RetrievalMetricV2(force_recalc_index=True)
        elif (args.output_file and 'hypermatch' in args.output_file) or (args.jsonl_file and 'hypermatch' in args.jsonl_file):
            metrics_dict["retrieval_v2"] = RetrievalMetricV2(force_recalc_index=True)
        elif (args.output_file and 'segnet' in args.output_file) or (args.jsonl_file and 'segnet' in args.jsonl_file):
            metrics_dict["retrieval_v2"] = RetrievalMetricV2(force_recalc_index=True)
        else:
            metrics_dict["retrieval_v2"] = RetrievalMetricV2()
        
    if "rouge" in metrics:
        from metrics import RougeMetric
        metrics_dict["rouge"] = RougeMetric()
      
    if "semantic_matching" in metrics:
        from metrics import SemanticMatchingMetric
        metrics_dict["semantic_matching"] = SemanticMatchingMetric()
        
    if "unieval" in metrics:
        from metrics import UniEvalMetric
        metrics_dict["unieval"] = UniEvalMetric()

    print("Reading input files...")
    ids = []
    all_inputs = []
    all_refs = []
    all_preds = []
    if args.jsonl_file is not None:
        with open(args.jsonl_file) as f:
            data = json.load(f)
            all_inputs.extend([item['source'] for item in data])
            all_refs.extend([item['target'] for item in data])
            all_preds.extend([item['predictions'] for item in data])
            ids.extend([item['id'] for item in data])

            # for line in f.readlines():
            #     data = json.loads(line)
            #     try:
            #         ids.append(data['id'])
            #     except:
            #         pass
            #     all_inputs.append(data['source'])
            #     all_refs.append(data['target'])
            #     all_preds.append(data['predictions'])
            #     if args.only_use_first_n and len(all_inputs) == args.only_use_first_n:
            #         break
    else:
        with open(args.output_file) as f:
            all_preds = [line.strip() for line in f.readlines()]
        with open(args.label_file) as f:
            all_refs = [line.strip() for line in f.readlines()]
        with open(args.input_file) as f:
            all_inputs = [line.strip() for line in f.readlines()]
        
        if args.only_use_first_n:
            assert len(all_preds) >= args.only_use_first_n and len(all_refs) >= args.only_use_first_n and len(all_inputs) >= args.only_use_first_n
            all_preds = all_preds[:args.only_use_first_n]
            all_refs = all_refs[:args.only_use_first_n]
            all_inputs = all_inputs[:args.only_use_first_n]
        else:
            assert len(all_preds) == len(all_refs) == len(all_inputs)
    
    if len(ids) == 0:
        ids = list(range(0, len(all_preds)))


    '''
    # =====================================
    # TOKENIZATION
    print("Preparing the input")
    references_delimited = None
    summaries_delimited = None
    if len(references) > 0:
        if isinstance(references[0], list):
            if "line_delimited" in toks_needed:
                references_delimited = ["\n".join(ref) for ref in references]
            if "space" in toks_needed:
                references_space = [" ".join(ref) for ref in references]
        elif args.eos is not None:
            if "line_delimited" not in toks_needed:
                raise ValueError('You provided a delimiter but are not using a metric which requires one.')
            if args.eos == "\n":
                references_delimited = [ref.split(args.eos) for ref in references]
            else:
                references_delimited = [f"{args.eos}\n".join(ref.split(args.eos)) for ref in references]
        elif "line_delimited" in toks_needed:
            references_delimited = references
        if "space" in toks_needed:
            references_space = references

    if isinstance(summaries[0], list):
        if "line_delimited" in toks_needed:
            summaries_delimited = ["\n".join(summ) for summ in summaries]
        if "space" in toks_needed:
            summaries_space = [" ".join(summ) for summ in summaries]
    elif args.eos is not None:
        if "line_delimited" not in toks_needed:
            raise ValueError('You provided a delimiter but are not using a metric which requires one.')
        if args.eos == "\n":
            summaries_delimited = [ref.split(args.eos) for ref in summaries]
        else:
            summaries_delimited = [f"{args.eos}\n".join(ref.split(args.eos)) for ref in summaries]
    elif "line_delimited" in toks_needed:
        summaries_delimited = summaries
    if "space" in toks_needed:
        summaries_space = summaries

    if "stem" in toks_needed:
        tokenizer = RegexpTokenizer(r'\w+')
        stemmer = Snmetrics_dictowballStemmer("english")
        if isinstance(summaries[0], list):
            summaries_stemmed = [[stemmer.stem(word) for word in tokenizer.tokenize(" ".join(summ))] for summ in summaries]
            references_stemmed = [[stemmer.stem(word) for word in tokenizer.tokenize(" ".join(ref))] for ref in references]
        else:
            summaries_stemmed = [[stemmer.stem(word) for word in tokenizer.tokenize(summ)] for summ in summaries]
            references_stemmed = [[stemmer.stem(word) for word in tokenizer.tokenize(ref)] for ref in references]
        summaries_stemmed = [" ".join(summ) for summ in summaries_stemmed]
        references_stemmed = [" ".join(ref) for ref in references_stemmed]

    if "spacy" in toks_needed:
        try:
            nlp = spacy.load('en_core_web_md')
        except OSError:
            print('Downloading the spacy en_core_web_md model\n'
                "(don't worry, this will only happen once)", file=stderr)
            from spacy.cli import download
            download('en_core_web_md')
            nlp = spacy.load('en_core_web_md')
        disable = ["tagger", "textcat"]
        if "summaqa" not in metrics:
            disable.append("ner")
        if isinstance(summaries[0], list):
            summaries_spacy = [nlp(" ".join(text), disable=disable) for text in summaries]
        else:
            summaries_spacy = [nlp(text, disable=disable) for text in summaries]
        if "stats" in metrics:
            summaries_spacy_stats = [[tok.text for tok in summary] for summary in summaries_spacy]
        if "sms" in metrics:
            if isinstance(references[0], list):
                references_spacy = [nlp(" ".join(text), disable=disable) for text in references]
            else:
                references_spacy = [nlp(text, disable=disable) for text in references]
        if "summaqa" in metrics or "stats" in metrics:
            if isinstance(articles[0], list):
                input_spacy = [nlp(" ".join(text), disable=disable) for text in articles]
            else:
                input_spacy = [nlp(text, disable=disable) for text in articles]
            if "stats" in metrics:
                input_spacy_stats = [[tok.text for tok in article] for article in input_spacy]
    if "supert" in metrics or "blanc" in metrics:
        inputs_space = articles
    # =====================================
    '''

    
    # =====================================
    # GET SCORES
    final_output_aggregated, final_output_separated = {}, {}

    '''
    if args.aggregate:
        final_output = dict()
    else:
        final_output = defaultdict(lambda: defaultdict(int))
    #import pdb;pdb.set_trace()
    '''
    for metric, metric_cls in metrics_dict.items():
        print("================================================================================")
        print(f"Calculating scores for the {metric} metric.")
        per_doc_scores, aggregated_scores = metric_cls.evaluate(all_preds, all_refs, all_inputs)
        print(aggregated_scores)
        final_output_aggregated[metric] = aggregated_scores
        final_output_separated[metric] = per_doc_scores

    
    '''
    for metric, metric_cls in metrics_dict.items():
        print(f"Calculating scores for the {metric} metric.")
        try:
            if metric == "rouge":
                output = metric_cls.evaluate_batch(summaries_delimited, references_delimited, aggregate=args.aggregate)
                # only rouge uses this input so we can delete it
                del references_delimited
                del summaries_delimited
            elif metric in ('bert_score', 'mover_score', 'chrf', 'meteor', 'bleu'):
                output = metric_cls.evaluate_batch(summaries_space, references_space, aggregate=args.aggregate)
            elif metric in ('s3', 'rouge_we', 'cider'):
                output = metric_cls.evaluate_batch(summaries_stemmed, references_stemmed, aggregate=args.aggregate)
            elif metric == "sms":
                output = metric_cls.evaluate_batch(summaries_spacy, references_spacy, aggregate=args.aggregate)
            elif metric in ('summaqa', 'stats', 'supert', 'blanc'):
                if metric == "summaqa":
                    output = metric_cls.evaluate_batch(summaries_space, input_spacy, aggregate=args.aggregate)
                elif metric == "stats":
                    output = metric_cls.evaluate_batch(summaries_spacy_stats, input_spacy_stats, aggregate=args.aggregate)
                elif metric in ('supert', 'blanc'):
                    output = metric_cls.evaluate_batch(summaries_space, inputs_space, aggregate=args.aggregate)
            if args.aggregate:
                final_output.update(output)
            else:
                ids = list(range(0, len(ids)))
                for cur_id, cur_output in zip(ids, output):
                    final_output[cur_id].update(cur_output)
        except Exception as e:
            print(e)
            print(f"An error was encountered with the {metric} metric.")
    # =====================================
    '''

    
    # =====================================
    # OUTPUT SCORES
    metrics_str = "_".join(metrics)
    metrics_str_abbrv = "_".join([metric[:4] for metric in metrics])
    # if args.only_use_first_n:
    #     args.log_file_prefix = args.log_file_prefix + f'_first{args.only_use_first_n}docs'
    
    if args.log_file_prefix[-1] == '/' or os.path.isdir(args.log_file_prefix):
        out_file_aggregated  = f"{args.log_file_prefix}/{metrics_str_abbrv}_aggregated_{args.jsonl_file.split('/')[-1]}"
        # out_file_perdoc = f"{args.log_file_prefix}/{metrics_str_abbrv}_perdoc.jsonl"
    else:
        out_file_aggregated  = f"{args.log_file_prefix}_{metrics_str_abbrv}_aggregated_{args.jsonl_file.split('/')[-1]}"
        # out_file_perdoc = f"{args.log_file_prefix}_{metrics_str_abbrv}_perdoc.jsonl"
    
    with open(out_file_aggregated, "w") as outputf:
        json.dump(final_output_aggregated, outputf, cls=NpEncoder, indent=4)
    # with open(out_file_perdoc, "w") as outputf:
    #     json.dump(final_output_separated, outputf, cls=NpEncoder)


if __name__ == '__main__':
    cli_main()