
def display_hash(hashTable):
    
    for i in range(len(hashTable)):
        print(i, end = " ")
        
        for j in hashTable[i]:
            print("-->", end = " ")
            print(j, end = " ")
        print()
        

HashTable = [[] for _ in range(10)]

def Hashing(key):
    return key % len(HashTable)

def insert(Hashtable, key, value):
    hash_key = Hashing(key)
    Hashtable[hash_key].append(value)
    
insert(HashTable, 20, 'Dario')
insert(HashTable, 20, 'Rose')
display_hash (HashTable)