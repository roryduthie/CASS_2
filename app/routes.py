from flask import render_template, request, redirect, session, Markup
from . import app
import pandas as pd
from urllib.request import urlopen
import requests
import json
from app.aifsim import Aifsim
from pycm import *

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
    text1 = aifs.get_text(fid)
    text2 = aifs.get_text(lid)

    ss, pk, win_diff = aifs.get_similarity(text1, text2)
    g, g_json, g1, g1_json = aifs.get_graphs(fid,lid)
    prop_rels = aifs.get_prop_sim_matrix(g, g1)
    loc_rels = aifs.get_loc_sim_matrix(g, g1)

    ra_a = aifs.ra_anchor(g, g1)
    ma_a = aifs.ma_anchor(g, g1)
    ca_a = aifs.ca_anchor(g, g1)
    all_a = aifs.combine_s_node_matrix(ra_a, ca_a, ma_a)
    all_s_a_dict = aifs.convert_to_dict(all_a)

    prop_rels_comp_conf = aifs.prop_rels_comp(prop_rels, g, g1)
    prop_rels_comp_dict = aifs.convert_to_dict(prop_rels_comp_conf)

    loc_ya_rels_comp_conf = aifs.loc_ya_rels_comp(loc_rels, g, g1)
    loc_ya_rels_comp_dict = aifs.convert_to_dict(loc_ya_rels_comp_conf)

    prop_ya_comp_conf = aifs.prop_ya_comp(prop_rels, g, g1)
    prop_ya_comp_dict = aifs.convert_to_dict(prop_ya_comp_conf)

    loc_ta_conf = aifs.loc_ta_rels_comp(loc_rels, g, g1)
    loc_ta_dict = aifs.convert_to_dict(loc_ta_conf)

    prop_ya_conf = aifs.prop_ya_anchor_comp(prop_rels, g, g1)
    prop_ya_dict = aifs.convert_to_dict(prop_ya_conf)



    all_s_a_cm = ConfusionMatrix(matrix=all_s_a_dict)
    prop_rels_comp_cm = ConfusionMatrix(matrix=prop_rels_comp_dict)
    loc_ya_rels_comp_cm = ConfusionMatrix(matrix=loc_ya_rels_comp_dict)
    prop_ya_comp_cm = ConfusionMatrix(matrix=prop_ya_comp_dict)
    loc_ta_cm = ConfusionMatrix(matrix=loc_ta_dict)
    prop_ya_cm = ConfusionMatrix(matrix=prop_ya_dict)



    s_node_kapp = all_s_a_cm.Kappa
    prop_rel_kapp = prop_rels_comp_cm.Kappa
    loc_rel_kapp = loc_ya_rels_comp_cm.Kappa
    prop_ya_kapp = prop_ya_comp_cm.Kappa
    loc_ta_kapp = loc_ta_cm.Kappa
    prop_ya_an_kapp = prop_ya_cm.Kappa

    if aifs.check_none(s_node_kapp):
        s_node_kapp = all_s_a_cm.KappaNoPrevalence
    if aifs.check_none(prop_rel_kapp):
        prop_rel_kapp = prop_rels_comp_cm.KappaNoPrevalence
    if aifs.check_none(loc_rel_kapp):
        loc_rel_kapp = loc_ya_rels_comp_cm.KappaNoPrevalence
    if aifs.check_none(prop_ya_kapp):
        prop_ya_kapp = prop_ya_comp_cm.KappaNoPrevalence
    if aifs.check_none(loc_ta_kapp):
        loc_ta_kapp = loc_ta_cm.KappaNoPrevalence
    if aifs.check_none(prop_ya_an_kapp):
        prop_ya_an_kapp = prop_ya_cm.KappaNoPrevalence

    score_list = [s_node_kapp,prop_rel_kapp,loc_rel_kapp,prop_ya_kapp,loc_ta_kapp,prop_ya_an_kapp]

    print(score_list)

    k_graph = sum(score_list) / float(len(score_list))

    graph_sim = k_graph
    text_sim = ss

    overall_sim = (graph_sim + text_sim) / 2

    return render_template('results.html', overall=overall_sim, text_sim=text_sim, graph_sim=graph_sim)
    #return render_template('results.html', overall, table=[items])

