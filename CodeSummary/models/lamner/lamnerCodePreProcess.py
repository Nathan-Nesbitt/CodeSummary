import javalang
import re 
from datetime import datetime
from spiral import ronin


def make_str(st):
  new_s = ""
  rom =False
  for char in st:
    char = str(char)
    if char.isupper() and len(str(char))==1:
      if rom==True:
        new_s=new_s + " "
        rom = False
      new_s = new_s+char
      new_s = new_s.strip()
    else:
      rom=True
      new_s = new_s + " " + char
      new_s = new_s.strip()
  return new_s


def camel_case_split(str):
  ac = ronin.split(str)
  a = " ".join(ac)
  return a
  
  
def get_lm_embeds(code_snippet, indexes, processed_final_code):
  intermediate_tokens =[]
  splitted_code = code_snippet.split(" ")
  
  for i in indexes:
    intermediate_tokens.append(splitted_code[i])
  
  prc_code = ""
  for i in range(len(intermediate_tokens)):
    rem_num_str = intermediate_tokens[i]
    rem_num_str = rem_num_str.replace("_", "")
    rem_num_str = rem_num_str.replace("@", "")
    rem_num_str = rem_num_str.replace("0", "")
    rem_num_str = rem_num_str.replace("1", "")
    rem_num_str = rem_num_str.replace("2", "")
    rem_num_str = rem_num_str.replace("3", "")
    rem_num_str = rem_num_str.replace("4", "")
    rem_num_str = rem_num_str.replace("5", "")
    rem_num_str = rem_num_str.replace("6", "")
    rem_num_str = rem_num_str.replace("7", "")
    rem_num_str = rem_num_str.replace("8", "")
    rem_num_str = rem_num_str.replace("9", "")



    regexed_string = camel_case_split(rem_num_str)
    
    final_regexed_string = " ".join(regexed_string.split(" ")).strip(" ")
    splitted_regexed_string = final_regexed_string.split()
    prc_code = prc_code + " " +" ".join(splitted_regexed_string).strip(" ")
    prc_code = prc_code.strip(" ").lower()
  return prc_code
  
  
  
def tokenize_code(code_snippet):
  indexes = []
  tree = list(javalang.tokenizer.tokenize(code_snippet))
  s_lm= ""   # slm is string which contains complete language model representation with num str
  processed_final_code = "" # full code but rencos processed
  for i in range(len(tree)):

    # j[0] is token type, and j[1] is token
    j = str(tree[i])
    j = j.split()
    
    if "decimalinteger" in j[0].lower() or "decimalfloatingpoint" in j[0].lower():
      j[0] = "NUM"
      j[1] = "NUM"
    j[1] = j[1].strip('"')

    if "string"==j[0].lower():
      j[0] = "STR"
      j[1] = "STR"
    j[1] = j[1].strip('"')
    
    s_lm =  s_lm + " " + j[1].strip('"')
    s_lm= s_lm.strip(" ")

    if "separator"==j[0].lower() or "operator"==j[0].lower() :
      continue
    
    indexes.append(i)
    processed_final_code =  processed_final_code + " " + j[1].strip('"')
    processed_final_code = processed_final_code.strip(" ") 

  #return s_lm, processed_final_code, indexes
  return get_lm_embeds(s_lm, indexes, processed_final_code)