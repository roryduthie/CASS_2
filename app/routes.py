from flask import render_template, request, redirect, session, Markup
from . import app
import pandas as pd
from urllib.request import urlopen
import requests
import json
from app.aifsim import Aifsim

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/', methods=['POST'])
def form_post():
    fid = request.form['fdata']
    lid = request.form['ldata']
    session['fid'] = fid
    session['lid'] = lid
    return redirect('/results')

@app.route('/index', methods=['POST'])
def form_pos():
    fid = request.form['fdata']
    lid = request.form['ldata']
    session['fid'] = fid
    session['lid'] = lid
    return redirect('/results')

@app.route('/results')
def render_text():
    fid = session.get('fid', None)
    lid = session.get('lid', None)
    aifs = Aifsim()
    res = aifs.get_graph_sim(fid,lid)
    overall_sim = res[0]
    text_sim = res[1]
    graph_sim = res[2]
    return render_template('results.html', overall=overall_sim, text_sim=text_sim, graph_sim=graph_sim)
    #return render_template('results.html', overall, table=[items])

