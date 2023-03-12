def isSubstring(s1,s2):
    if s1.find(s2) != -1:
        return True
    else:
        return False

def isRotation(s1,s2):
    lengths1 = len(s1)
    if(lengths1 == len(s2) and lengths1 > 0):
        return (s1 + s1, s2)
    else:
        return False

if __name__ == '__main__':
    result = isRotation('enabler', 'erenabl7')
    if result:
        print('yes, its a rotation!')
    else:
        print('no, its not a rotation!')
    
