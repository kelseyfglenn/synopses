from itertools import chain

def clean_generated_file(text):
    """
    take in a generated text file (single string containing multiple samples in mutliple batches)
    split into individual strings and remove generator artifact tokens

    input format (single string):
    
    'texttexttext<|endoftext|>
    <|startoftext|>textexttext<|endoftext|>
    ...
    \n====================\n
    texttexttext<|endoftext|>
    <|startoftext|>textexttext<|endoftext|>
    ...'
    
    input: string (file.read())
    returns: list of strings (separated and cleaned)
    """
 
    # there's some messiness in the endoftext/startoftext tokens that makes splitting on them inconsistent
    # so after some experimenting this approach seems to work better
    
    # split on < and > from '<|startoftext|>' and '<|endoftext|>' tokens
    text = text.split('<') # str -> ['str', 'str',...]
    text = [x.split('>') for x in text] # [['str'], ['str', 'str'], ...]
    text = list(chain(*text)) # flatten list ['str', 'str',...]
    # split elements on batch separator token
    text = [x.split('\n====================\n') for x in text] 
    text = list(chain(*text)) # flatten list
    # remove generator token artifacts
    rem_tokens = ['|startoftext|', '|endoftext|', '\n', '']
    text = [x for x in text if x not in rem_tokens]
    text = [x for x in text if '|' not in x] # catch tokens that somehow got split
    text = [x[1:] if x[0]=='\n' else x for x in text] # remove initial newlines

    # return list of cleaned texts
    return text