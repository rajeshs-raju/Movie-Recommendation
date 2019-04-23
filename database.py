import sqlite3
conn = sqlite3.connect('database.db')
print("Opened database successfully");
#conn.execute('CREATE TABLE users (ID INTEGER PRIMARY KEY AUTOINCREMENT,uname TEXT, emailid TEXT, password TEXT, genre TEXT , recommended TEXT,hybrid_recommendation TEXT,liked_movies TEXT)')
conn.execute("INSERT INTO users (ID,uname,emailid,password,genre,recommended,liked_movies)  VALUES (?,?,?,?,?,?,?)",(672,'Rajesh','rajraj@pes.edu','123456','Adventure,Crime','','') )
conn.commit()
print("Table created successfully");
conn.close()