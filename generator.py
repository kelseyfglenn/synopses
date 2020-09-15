import tensorflow
import gpt_2_simple as gpt2
from clean_generations import clean_generated_file

sess=None

def generate_from_seed(seed, session=sess, n_samples=1, max_len=200, temp=0.7):
    """
    generate n samples from a starting seed
    """
    gen = gpt2.generate(sess,
              length=max_len,
              temperature=temp,
              nsamples=n_samples,
              batch_size=1,
              prefix=seed,
              return_as_list=True
              )
    return gen[0]

def generate_clean_sample(seed, session=sess, max_len):
    """
    generate a single sample that reached the end of text token
    """
    gen = generate_from_seed(seed, session=sess, max_len=max_len)
    clean = clean_generated_file(gen)
    return clean[0]