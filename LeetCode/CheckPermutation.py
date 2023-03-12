def ispermutation(s1,s2):
    s1_list = list()
    s2_list = list()
    s1_list.extend(s1)
    s2_list.extend(s2)
    s1_list.sort()
    s2_list.sort()
    if s1_list == s2_list:
      return True
    else:
      return False

if __name__ == "__main__":
  print(ispermutation('asd3','dsa'))
  
