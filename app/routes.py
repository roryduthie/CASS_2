from flask import render_template, request, redirect, session, Markup
from . import app
import pandas as pd
from urllib.request import urlopen
import requests
import json
from aifsim import Aifsim

@application.route('/')
@application.route('/index')
def index():
    return render_template('index.html')
@application.route('/index', methods=['POST'])
def form_post():
    fid = request.form['fdata']
    lid = request.form['ldata']
    session['fid'] = fid
    session['lid'] = lid
    return redirect('/results')

@application.route('/results')
def render_text():
    fid = session.get('fid', None)
    lid = session.get('lid', None)
    aifs = Aifsim()
    print(aifs.get_graph_sim(fid,lid))
    return render_template('')
    #return render_template('results.html', title=text, table=[items])
