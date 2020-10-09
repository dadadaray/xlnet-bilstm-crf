# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unicodedata
import six
from functools import partial
import sentencepiece as spm
#from xlnet import prepro_utils


SPIECE_UNDERLINE = '▁'


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s", (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")

def print_(*args):
  new_args = []
  for arg in args:
    if isinstance(arg, list):
      s = [printable_text(i) for i in arg]
      s = ' '.join(s)
      new_args.append(s)
    else:
      new_args.append(printable_text(arg))
  print(*new_args)


def preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
  if remove_space:
    outputs = ' '.join(inputs.strip().split())
  else:
    outputs = inputs
  outputs = outputs.replace("``", '"').replace("''", '"')

  if six.PY2 and isinstance(outputs, str):
    outputs = outputs.decode('utf-8')

  if not keep_accents:
    outputs = unicodedata.normalize('NFKD', outputs)
    outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
  if lower:
    outputs = outputs.lower()

  return outputs


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
  # return_unicode is used only for py2
  #print("encode_pieces:",text)
  # note(zhiliny): in some systems, sentencepiece only accepts str for py2
  if six.PY2 and isinstance(text, unicode):
    #print("this is python2")
    text = text.encode('utf-8')

  if not sample:
    pieces = sp_model.EncodeAsPieces(text)


  else:
    pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    #print("2encode_pieces_sample_piece", pieces)
  new_pieces = []
  for piece in pieces:
    if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
      cur_pieces = sp_model.EncodeAsPieces(
          piece[:-1].replace(SPIECE_UNDERLINE, ''))
      if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
        if len(cur_pieces[0]) == 1:
          cur_pieces = cur_pieces[1:]
        else:
          cur_pieces[0] = cur_pieces[0][1:]
      cur_pieces.append(piece[-1])
      new_pieces.extend(cur_pieces)
    else:
      new_pieces.append(piece)

  # note(zhiliny): convert back to unicode for py2
  if six.PY2 and return_unicode:
    ret_pieces = []
    for piece in new_pieces:
      if isinstance(piece, str):
        piece = piece.decode('utf-8')
      ret_pieces.append(piece)
    new_pieces = ret_pieces

  return new_pieces


def encode_ids(sp_model, token, sample=False):
  #pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
  ids = [sp_model.PieceToId(piece) for piece in token]
  return ids

class FullTokenizer(object):

  def __init__(self, sp_model_file,lower_case=False):
    self.sp_processor=spm.SentencePieceProcessor()
    self.sp_processor.Load(sp_model_file)
    self.lower_case = lower_case

  def tokenize(self, text):
    #text=convert_to_unicode(text)
    #print("prepro_utils_text:",text)
    """Tokenize text for XLNet"""
    processed_text = preprocess_text(text, lower=self.lower_case)
    #print("prepro_util_processed_text",processed_text)
    #prepro_func = partial(preprocess_text, lower=True)
    tokenized_pieces = encode_pieces(self.sp_processor,processed_text)
    #print("prepro_utils_token",tokenized_pieces)
    #tokenized_pieces=convert_to_unicode(tokenized_pieces)

    return tokenized_pieces

  def encode(self, text):
    """Encode text for XLNet"""
    #text = convert_to_unicode(text)
    processed_text = preprocess_text(text, lower=self.lower_case)
    encoded_ids =encode_ids(self.sp_processor, processed_text)
    return encoded_ids

  def convert_tokens_to_ids(self,token):
    #token = convert_to_unicode(token)
    return self.sp_processor.PieceToId(token)

  def id_to_token(self,
                  id):
    """Convert id to token for XLNet"""
    return self.sp_processor.IdToPiece(id)


  def tokens_to_ids(self,
                  tokens):
    """Convert tokens to ids for XLNet"""
    #tokens = convert_to_unicode(tokens)
    return [self.sp_processor.PieceToId(token) for token in tokens]


  def ids_to_tokens(self,
                  ids):
    """Convert ids to tokens for XLNet"""
    return [self.sp_processor.IdToPiece(id) for id in ids]



if __name__ == '__main__':
  text="I was borner in 92000, and this is falsé. the students are stupid."
  t=FullTokenizer('../xlnet_cased_L-12_H-768_A-12/spiece.model')
  t1=t.tokenize(text)
  #t2=t.convert_tokens_to_ids(text)
  print(t1)
  #print(t2)







  import sentencepiece as spm
  #
  sp = spm.SentencePieceProcessor()
  sp.load('../xlnet_cased_L-12_H-768_A-12/spiece.model')
  #
  print_(u'I was born in 2000, and this is falsé.')
  print_(u'ORIGINAL', sp.EncodeAsPieces(u'I was born in 2000, and this is falsé.'))
  print_('OURS+++', encode_pieces(sp, u'I was born in 2000, and this is falsé.'))
  print(encode_ids(sp, u'I was born in 2000, and this is falsé.'))
  print_('')
  prepro_func = partial(preprocess_text, lower=True)
  print_("prepro_func",prepro_func('I was born in 2000, and this is falsé.'))
  print_('ORIGINAL', sp.EncodeAsPieces(prepro_func('I was born in 2000, and this is falsé.')))
  print_('OURS', encode_pieces(sp, prepro_func('I was born in 2000, and this is falsé.')))
  print(encode_ids(sp, prepro_func('I was born in 2000, and this is falsé.')))
  print_('')
  print_('I was born in 2000, and this is falsé.')
  print_('ORIGINAL', sp.EncodeAsPieces('I was born in 2000, and this is falsé.'))
  print_('OURS', encode_pieces(sp, 'I was born in 2000, and this is falsé.'))
  print(encode_ids(sp, 'I was born in 2000, and this is falsé.'))
  print_('')
  print_('I was born in 92000, and this is falsé.')
  print_('ORIGINAL', sp.EncodeAsPieces('I was born in 92000, and this is falsé.'))
  print_('OURS', encode_pieces(sp, 'I was born in 92000, and this is falsé.'))
  print(encode_ids(sp, 'I was born in 92000, and this is falsé.'))
  print("中文增加",encode_pieces(sp, '啦啦啦I was born in 2000, and this is falsé.'))
  print("中文增加", encode_pieces(sp, printable_text('啦啦啦I was born in 2000, and this is falsé.')))

