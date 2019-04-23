import os
import re
import sqlite3 as sql
import csv

if __name__ == "__main__":

	'''

	with sql.connect("database.db") as con:

		cur = con.cursor()
		result=cur.execute("select ID from users where uname = ?",('Rajesh',))
		for row in result:
			result=(row[0])
		print(result,type(result))
		#print(l)
		#cur.execute("update users set recommended=? where uname=?",(l,'Rajesh',))

	'''

	writer = open('input/ratings_small.csv','a')
	writer.seek(0,2)
	writer.writelines("\r")
	writer.writelines( (',').join(['0','0','0','0']))