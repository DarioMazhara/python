import sqlite3

connection = sqlite3.connect("dm.db")

crsr = connection.cursor()

print("Connected to database")

#################################### Body ###################################################


crsr.execute('''CREATE TABLE PERSON(PK VARCHAR(255), FNAME VARCHAR(255), LNAME VARCHAR(255)''')
# Primary key
pk = [2, 3, 4, 5, 6]

# Enter 5 people f_name
f_name = ['Dario', 'Rose', 'Ethan', 'Krishan', 'Nick']

# Enter l_name
l_name = ['Mazhara', 'Bafaiz', 'Gary', 'Lall', 'Ostrovskiy']

gender = ['M', 'F', 'M', 'M', 'M']

join_data = ['2018-03-22', '2018-10-14', '2019-05-05', '2019-12-10', '2020-06-16']

for i in range(5):
    crsr.execute(f'INSERT INTO PERSON VALUES ({pk[i]}, "{f_name[i]}", "{l_name[i]}", "{gender[i]}", "{join_data[i]}")')

#############################################################################################

connection.commit()

connection.close()