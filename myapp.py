from flask import Flask, render_template,request,session,redirect,url_for
#from ipynb.fs.full.mr import * 
#from mr import *
from hybrid import build_chart,get_recommendations,hybrid
import sqlite3 as sql
import pandas as pd
import csv
import random

app = Flask(__name__)

'''
data = pd.read_csv('movies.csv')

movies_list = data['title'].tolist()
genre_list = data['genres'].tolist()

movies_dict = {}

for i in range(len(movies_list)):
	movies_dict[movies_list[i]] = genre_list[i]
'''

@app.route('/')
def login_page():

	return render_template('login.html')

@app.route('/home_page', methods=['GET','POST'])
def home_page():
	if(request.method == 'POST'):
		print("****************coming here*************************")
		uname = request.form['username']
		password = request.form['password']
		print(uname,password)

		#x=genre_recommendation('Adventure') #get genre from database here
		#extract genre here
		con = sql.connect("database.db")
		cur = con.cursor()
		cur.execute("select genre from users where uname=?",(uname,))
		#print(cur.fetchall())
		l=cur.fetchall()[0]
		user_genre_list=[]
		for i in l:
			user_genre_list.append(i)
		user_genre_list=(user_genre_list[0].split(','))
		
		#genre_recommendations('')
		
		print("##################")
		print(user_genre_list)
		print("###################")
		#l=genre_recommendations(m)
		l=build_chart(user_genre_list[0]).head(5)
		l3=l.to_dict()
		build_chart_list=[]
		title_list = []
		year_list = []
		id_list = []
		for i in l3['id'].values():
			id_list.append(i)
		for i in l3['title'].values():
			i=i.replace(' ','_')
			i=i.replace(':','_')
			title_list.append(i)
		for i in l3['year'].values():
			year_list.append(i)
		for i in range(len(title_list)):
			p = title_list[i]
			q = year_list[i]
			i = id_list[i]
			item = dict (Name = p ,year = q , id = i)
			build_chart_list.append(item)
		print("##########################")
		print(build_chart_list)
		print("##########################")

		con.close()
		session['username'] = request.form['username']
		print("Im L3 :",l3)

		return render_template('homepage.html',uname=uname,val=user_genre_list[0],val1=user_genre_list[1],x=build_chart_list)

	else:
		if 'username' in session:
			username = session['username']
			print(username)

		con = sql.connect("database.db")
		cur = con.cursor()
		cur.execute("select genre from users where uname=?",(session['username'],))
		#print(cur.fetchall())
		l=cur.fetchall()[0]
		user_genre_list=[]
		for i in l:
			user_genre_list.append(i)
		user_genre_list=(user_genre_list[0].split(','))
		
		#genre_recommendations('')

		
		print("##################")
		print(user_genre_list)
		print("###################")
		#l=genre_recommendations(m)
		l=build_chart(user_genre_list[0]).head(5)
		l3=l.to_dict()
		build_chart_list=[]
		title_list = []
		year_list = []
		id_list = []
		for i in l3['id'].values():
			id_list.append(i)
		for i in l3['title'].values():
			i=i.replace(' ','_')
			i=i.replace(':','_')
			title_list.append(i)
		for i in l3['year'].values():
			year_list.append(i)
		for i in range(len(title_list)):
			p = title_list[i]
			q = year_list[i]
			i = id_list[i]
			item = dict (Name = p ,year = q , id = i)
			build_chart_list.append(item)
		print("##########################")
		print(build_chart_list)
		print("##########################")
		'''
		result_list=[]
		for i in l:
			result_list.append(i[:-6])
		'''

		con.close()

		return render_template('homepage.html',uname=session['username'],val=user_genre_list[0],val1=user_genre_list[1],x=build_chart_list)

@app.route('/recommender_page')
def recommender_page():
	'''
	uname = request.form['username']
	password = request.form['password']
	print(uname,password)
	return render_template('homepage.html',uname=uname)
	'''
	if 'username' in session:
		username = session['username']
	with sql.connect("database.db") as con:
		cur = con.cursor()
		result=cur.execute("select recommended from users where uname = ?",(username,))
		#hybrid_out=cur.execute("select hybrid_recommendation from users where uname = ?",(username,))
		l=[]
		content_based = []
		for row in result:
			try:
				l=row[0].split(',')
			except:
				pass
		try:
			for i in l:
				item = dict(Name = i)
				content_based.append(item)
		except:
			pass
		print("@@@@@@@@@@@@@@@@@@@@@@@@")
		print(content_based)
		print("@@@@@@@@@@@@@@@@@@@@@@@@")

		hybrid_out=cur.execute("select hybrid_recommendation from users where uname = ?",(username,))
		hybrid_list = []
		h=[]
		for rows in hybrid_out:
			try:
				h = rows[0].split(',')
			except:
				pass
		try:
			for i in h:
				item = dict(Name = i)
				hybrid_list.append(item)
		except:
			pass


	return render_template('recommended.html',uname=username,x=content_based,y=hybrid_list)

@app.route('/popular_page')
def popular_page():
	return render_template('popular.html',uname='Rajesh')

@app.route('/insert_user_rating',methods=['POST'])
def insert_user_rating():
	movie_name = request.form['movie_name']
	movie_id = request.form['movie_id']
	ratings = request.form['rating']
	print("****************************")
	print(movie_name,movie_id,ratings)
	print("*****************************")
	if 'username' in session:
		username = session['username']
	with sql.connect("database.db") as con:
		cur = con.cursor()
		cur.execute("update users set liked_movies=? where uname=?",(movie_name,username,))
		try:
			movie_name = movie_name.replace('_',' ')

			movie_name = movie_name.lstrip()
			movie_name = movie_name.rstrip()
			new_list=get_recommendations(movie_name).head(5)
			new_list=new_list.tolist()
			result=cur.execute("select recommended from users where uname = ?",(username,))
			userid = cur.execute("select ID from users where uname = ?",(username,))
			for row in userid:
				userid = row[0]
			l=[]
			for row in result:
				l=row[0].split(',')
			l.extend(new_list)
			l=set(l)  #to remove duplicates if any
			l=list(l)
			l=','.join(l)
			##################
			new_list = ','.join(new_list)
			###################
			print("7777777777777777777")
			print(new_list)
			print(movie_name)
			print((l))
			print(userid,movie_id,ratings)
			print("88888888888888888888")
			cur.execute("update users set recommended=? where uname=?",(new_list,username,))
			writer = open('input/ratings_small.csv','a')
			writer.seek(0,2)
			writer.writelines("\r")
			writer.writelines( (',').join([str(userid),str(movie_id),str(ratings),'12']))

			hybrid_out = hybrid(userid,movie_name)
			title_list = []
			hybrid_out = hybrid_out.to_dict()
			for i in hybrid_out['title'].values():
				title_list.append(i)
			title_list = ','.join(title_list)
			cur.execute("update users set hybrid_recommendation=? where uname=?",(title_list,username,))


		except KeyError:
			print(len(movie_name))
			print(movie_name)
			print("Movie name given wrongly")

		con.commit()
		print("Inserted successfully")


	return redirect(url_for('home_page'))


@app.route('/insert_user',methods =['POST'])
def insert_user():
	uname = request.form['uname']
	email = request.form['email']
	pwd = request.form['psw-repeat']
	genres = request.form.getlist('genre')
	print(uname,email,pwd,genres)
	genre_str = ','.join(genres)
	with sql.connect("database.db") as con:
		print("Connected to database")
		cur = con.cursor()
		cur.execute("INSERT INTO users (uname,emailid,password,genre,recommended,liked_movies)  VALUES (?,?,?,?,?,?)",(uname,email,pwd,genre_str,'','') )
		con.commit()
		msg  = 'record inserted successfully'
		print(msg)
	return render_template('login.html')


@app.route('/sign_up')
def sign_up():

	return render_template('signup.html')

@app.route('/logout')
def logout():
   # remove the username from the session if it is there
   session.pop('username', None)
   return render_template('login.html')



if __name__ == '__main__':
	app.secret_key = 'super secret key'
	app.run(debug = True)