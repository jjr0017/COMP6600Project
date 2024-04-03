import argparse
import sys

ALL_MODELS = ['dt', 'nb', 'lr', 'svm', 'mlp']

def get_arg_parser():

    parser = argparse.ArgumentParser(description='Parse Command Line Arguments for COMP6600 Project.')
    parser.add_argument('-m', '--models', nargs='+',
                        choices=['decision tree', 'dt',
                                'naive bayes', 'nb',
                                'logistic regression', 'lr',
                                'support vector machine', 'svm',
                                'multi-layered perceptron', 'mlp',
                                'all'],
                        default=['all'], required=False, dest='models',
                        help='which models to run')
    parser.add_argument('-c', '--clear', help="Clear huggingface cache", required=False, 
                        action=argparse.BooleanOptionalAction, default=False)
    
    return parser

def parse_args(args):
    args = args[1:]
    parser = get_arg_parser()
    parsed_args = parser.parse_args(args)

    models = {'dt': False,
              'nb': False,
              'lr': False,
              'svm': False,
              'mlp': False}
    if 'all' in parsed_args.models:
        models['dt'] = True
        models['nb'] = True
        models['lr'] = True
        models['svm'] = True
        models['mlp'] = True
    if 'dt' in parsed_args.models or 'decision tree' in parsed_args.models:
        models['dt'] = True
    if 'nb' in parsed_args.models or 'naive bayes' in parsed_args.models:
        models['nb'] = True
    if 'lr' in parsed_args.models or 'logistic regression' in parsed_args.models:
        models['lr'] = True
    if 'svm' in parsed_args.models or 'support vector machine' in parsed_args.models:
        models['svm'] = True
    if 'mlp' in parsed_args.models or 'multi-layered perceptron' in parsed_args.models:
        models['mlp'] = True
    
    return models, parsed_args.clear
    
if __name__ == '__main__':
    models, clear = parse_args(sys.argv)

    print(models)
    print(clear)

