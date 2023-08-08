
import json 

test_cases_f = open("test_aspects.txt", 'w') 

with open('../dataset/Tweets_corenlp/test.json') as infile:
    all_data = []
    data = json.load(infile)
    for d in data:
        for aspect in d['aspects']: 
            text_list = list(d['token'])
            tok = list(d['token'])       # word token
            length = len(tok)            # real length
            tok = [t.lower() for t in tok]
            tok = ' '.join(tok)
            asp = list(aspect['term'])   # aspect
            asp = [a.lower() for a in asp]
            asp = ' '.join(asp)
            label = aspect['polarity']   # label
            pos = list(d['pos'])         # pos_tag 
            head = list(d['head'])       # head
            deprel = list(d['deprel'])   # deprel 
            aspect_post = [aspect['from'], aspect['to']] 

            post = [i-aspect['from'] for i in range(aspect['from'])] \
                    +[0 for _ in range(aspect['from'], aspect['to'])] \
                    +[i-aspect['to']+1 for i in range(aspect['to'], length)]
            # aspect mask
            if len(asp) == 0: 
                continue 
            else:
                mask = [0 for _ in range(aspect['from'])] \
                    +[1 for _ in range(aspect['from'], aspect['to'])] \
                    +[0 for _ in range(aspect['to'], length)]
            
            sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head,\
                        'deprel': deprel, 'length': length, 'label': label, 'mask': mask, \
                        'aspect_post': aspect_post, 'text_list': text_list}

            test_cases_f.write(asp+'\n') 
