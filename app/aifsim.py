from app.centrality import Centrality
import numpy as np
from numpy import unravel_index
import copy
import gmatch4py as gm
import networkx as nx
from fuzzywuzzy import fuzz
import re
import segeval
from bs4 import BeautifulSoup, SoupStrainer
import bs4


class Aifsim:
    @staticmethod
    def get_graph(aif_id, cent):
        dir_path = 'http://www.aifdb.org/json/' + str(aif_id)
        graph, json = cent.get_graph_url(dir_path)
        return graph, json

    @staticmethod
    def get_text(nodeset_id):
        text_path = 'http://ova.arg.tech/helpers/dbtxt.php?nodeSetID=' + str(nodeset_id)
        xml_page = requests.get(text_path)
        xml_data = xml_page.text
        return xml_data

    @staticmethod
    def get_similarity(text_1, text_2):
    #text_1 and text_2 are xml data that uses spans to seperate boundaries
    #e.g. BOSTON, MA ... <span class="highlighted" id="634541">Steven L.
    #Davis pled guilty yesterday to federal charges that he stole and disclosed trade secrets of The Gillette Company</span>.

        if text_1 == '' or text_2 == '':
            return 'Error Text Input Is Empty'
        else:

            xml_soup_1 = BeautifulSoup(text_1)
            xml_soup_2 = BeautifulSoup(text_2)
            xml_soup_1 = remove_html_tags(xml_soup_1)
            xml_soup_2 = remove_html_tags(xml_soup_2)

            segements_1 = get_segements(xml_soup_1)
            segements_2 = get_segements(xml_soup_2)

            seg_check = check_segment_length(segements_1, segements_2)

            if not seg_check:
                return 'Error Source Text Was Different'

            masses_1 = segeval.convert_positions_to_masses(segements_1)
            masses_2 = segeval.convert_positions_to_masses(segements_2)

            ss = segeval.segmentation_similarity(masses_1, masses_2)
            ss = float(ss)
            pk = segeval.pk(masses_1, masses_2)
            pk = 1 - float(pk)
            win_diff = segeval.window_diff(masses_1, masses_2)
            win_diff = 1 - float(win_diff)

            return ss, pk, win_diff


    @staticmethod
    def is_iat(g, g1, centra):
        l_nodes = centra.get_l_node_list(g)
        l1_nodes = centra.get_l_node_list(g1)

        if len(l_nodes) < 1 and len(l1_nodes) < 1:
            return 'aif'
        elif len(l_nodes) > 1 and len(l1_nodes) > 1:
            return 'iat'
        else:
            return 'diff'

    @staticmethod
    def get_normalized_edit_distance(g1, g2, label_equal, attr_name):
        if label_equal:
            dist = nx.algorithms.similarity.optimize_graph_edit_distance(g1, g2, node_match=lambda a,b: a[attr_name] == b[attr_name])
        else:
            dist = nx.algorithms.similarity.optimize_graph_edit_distance(g1, g2)

        max_g_len = max(len(g1.nodes),len(g2.nodes))
        ed_dist = min(list(dist))


        norm_ed_dist = (max_g_len - ed_dist) / max_g_len

        return norm_ed_dist


    @staticmethod
    def get_normalized_path_edit_distance(g1, g2, label_equal, attr_name):
        if label_equal:
            dist = nx.algorithms.similarity.optimize_edit_paths(g1, g2, node_match=lambda a,b: a[attr_name] == b[attr_name])
        else:
            dist = nx.algorithms.similarity.optimize_edit_paths(g1, g2)

        max_g_len = max(len(g1.nodes),len(g2.nodes))
        ed_dist = min(list(dist))


        norm_ed_dist = (max_g_len - ed_dist) / max_g_len

        return norm_ed_dist


    @staticmethod
    def get_s_nodes(g):
        s_nodes = [x for x,y in g.nodes(data=True) if y['type']=='RA' or y['type']=='CA' or y['type']=='MA' or y['type']=='PA']
        not_s_nodes = [x for x,y in g.nodes(data=True) if y['type']!='RA' and y['type']!='CA' and y['type']!='MA' and y['type']!='PA']
        return s_nodes, not_s_nodes


    @staticmethod
    def get_i_s_nodes(g):
        i_s_nodes = [x for x,y in g.nodes(data=True) if y['type']=='I' or y['type']=='RA' or y['type']=='CA' or y['type']=='MA' or y['type']=='PA']
        not_i_s_nodes = [x for x,y in g.nodes(data=True) if y['type']!='I' and y['type']!='RA' and y['type']!='CA' and y['type']!='MA' and y['type']!='PA']
        return i_s_nodes, not_i_s_nodes


    @staticmethod
    def get_l_nodes(g):
        l_nodes = [x for x,y in g.nodes(data=True) if y['type']=='L']
        not_l_nodes = [x for x,y in g.nodes(data=True) if y['type']!='L']
        return l_nodes, not_l_nodes


    @staticmethod
    def get_l_ta_nodes(g):
        l_ta_nodes = [x for x,y in g.nodes(data=True) if y['type']=='L' or y['type']=='TA']
        not_l_ta_nodes = [x for x,y in g.nodes(data=True) if y['type']!='L' and y['type']!='TA']
        return l_ta_nodes, not_l_ta_nodes


    @staticmethod
    def get_i_nodes(g):
        i_nodes = [x for x,y in g.nodes(data=True) if y['type']=='I']
        not_i_nodes = [x for x,y in g.nodes(data=True) if y['type']!='I']
        return i_nodes, not_i_nodes


    @staticmethod
    def get_ya_nodes(g):
        ya_nodes = [x for x,y in g.nodes(data=True) if y['type']=='YA']
        not_ya_nodes = [x for x,y in g.nodes(data=True) if y['type']!='YA']
        return ya_nodes, not_ya_nodes


    @staticmethod
    def get_ta_nodes(g):
        ta_nodes = [x for x,y in g.nodes(data=True) if y['type']=='TA']
        not_ta_nodes = [x for x,y in g.nodes(data=True) if y['type']!='TA']
        return ta_nodes, not_ta_nodes


    @staticmethod
    def get_l_ya_nodes(g):
        l_ya_nodes = [x for x,y in g.nodes(data=True) if y['type']=='L' or y['type']=='YA']
        not_l_ya_nodes = [x for x,y in g.nodes(data=True) if y['type']!='L' and y['type']!='YA']
        return l_ya_nodes, not_l_ya_nodes


    @staticmethod
    def get_l_i_ya_nodes(g):
        l_i_ya_nodes = [x for x,y in g.nodes(data=True) if y['type']=='L' or y['type']=='YA' or y['type']=='I']
        not_l_i_ya_nodes = [x for x,y in g.nodes(data=True) if y['type']!='L' and y['type']!='YA' and y['type']!='I']
        return l_i_ya_nodes, not_l_i_ya_nodes


    @staticmethod
    def get_l_ta_ya_nodes(g):
        l_ta_ya_nodes = [x for x,y in g.nodes(data=True) if y['type']=='L' or y['type']=='YA' or y['type']=='TA']
        not_l_ta_ya_nodes = [x for x,y in g.nodes(data=True) if y['type']!='L' and y['type']!='YA' and y['type']!='TA']
        return l_ta_ya_nodes, not_l_ta_ya_nodes


    @staticmethod
    def get_i_s_ya_nodes(g):
        i_s_ya_nodes = [x for x,y in g.nodes(data=True) if y['type']=='I' or y['type']=='RA' or y['type']=='CA' or y['type']=='MA' or y['type']=='PA' or y['type']=='YA']
        not_i_s_ya_nodes = [x for x,y in g.nodes(data=True) if y['type']!='I' and y['type']!='RA' and y['type']!='CA' and y['type']!='MA' and y['type']!='PA' and y['type']!='YA']
        return i_s_ya_nodes, not_i_s_ya_nodes


    @staticmethod
    def remove_nodes(graph, remove_list):
        graph.remove_nodes_from(remove_list)
        return graph


    @staticmethod
    def get_i_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = Aifsim()
        g1_i_nodes, g1_not_i_nodes = aifsim.get_i_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_i_nodes)
        g2_i_nodes, g2_not_i_nodes = aifsim.get_i_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_i_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, False, '')
        return ed


    @staticmethod
    def get_s_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = Aifsim()
        g1_nodes, g1_not_nodes = aifsim.get_s_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_s_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, False, '')
        return ed


    @staticmethod
    def get_i_s_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = Aifsim()
        g1_nodes, g1_not_nodes = aifsim.get_i_s_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_i_s_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, True, 'type')
        return ed


    @staticmethod
    def get_l_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = Aifsim()
        g1_nodes, g1_not_nodes = aifsim.get_l_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_l_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, False, 'type')
        return ed


    @staticmethod
    def get_l_ta_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = Aifsim()
        g1_nodes, g1_not_nodes = aifsim.get_l_ta_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_l_ta_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, False, 'type')
        return ed


    @staticmethod
    def get_ya_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = Aifsim()

        g1_nodes, g1_not_nodes = aifsim.get_ya_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_ya_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, True, 'text')
        return ed


    @staticmethod
    def get_ya_l_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = Aifsim()

        g1_nodes, g1_not_nodes = aifsim.get_l_ya_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_l_ya_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, True, 'text')
        return ed


    @staticmethod
    def get_ta_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = Aifsim()

        g1_nodes, g1_not_nodes = aifsim.get_ta_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_ta_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, True, 'text')
        return ed


    @staticmethod
    def get_ya_l_i_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = Aifsim()

        g1_nodes, g1_not_nodes = aifsim.get_l_i_ya_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_l_i_ya_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, True, 'text')
        return ed


    @staticmethod
    def get_l_ta_ya_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = Aifsim()

        g1_nodes, g1_not_nodes = aifsim.get_l_ta_ya_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_l_ta_ya_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, False, 'type')
        return ed


    @staticmethod
    def get_i_s_ya_node_sim(g1, g2):
        g1_c = g1.copy()
        g2_c = g2.copy()
        aifsim = Aifsim()

        g1_nodes, g1_not_nodes = aifsim.get_i_s_ya_nodes(g1_c)
        new_g1 = aifsim.remove_nodes(g1_c, g1_not_nodes)
        g2_nodes, g2_not_nodes = aifsim.get_i_s_ya_nodes(g2_c)
        new_g2 = aifsim.remove_nodes(g2_c, g2_not_nodes)
        ed = aifsim.get_normalized_gm_edit_distance(new_g1, new_g2, True, 'type')
        return ed


    @staticmethod
    def findMean(a, N):

        summ = 0

        # total sum calculation of matrix
        for i in range(N):
            for j in range(N):
                summ += a[i][j]

        return summ/(N*N)


    @staticmethod
    def get_normalized_gm_edit_distance(g1, g2, label_equal, attr_name):
        ged=gm.GraphEditDistance(1,1,1,1)

        if label_equal:
            ged.set_attr_graph_used(attr_name, None)
            result=ged.compare([g1,g2],None)
        else:
            result=ged.compare([g1,g2],None)

        sim = ged.similarity(result)
        flat_sim = sim.flatten()
        flat_sim = flat_sim[flat_sim!=0]
        #print(flat_sim)
        norm_ed_dist = min(flat_sim)
        #norm_ed_dist = findMean(sim, 2)
        #norm_ed_dist = (sim[0][1] + sim[1][0])/2
        #print(sim)
        return norm_ed_dist


    @staticmethod
    def call_diagram_parts_and_sum(g_copy, g1_copy, rep):
        aifsim = Aifsim()
        if rep == 'aif':
            i_sim = aifsim.get_i_node_sim(g_copy, g1_copy)
            s_sim = aifsim.get_s_node_sim(g_copy, g1_copy)
            i_s_sim = aifsim.get_i_s_node_sim(g_copy, g1_copy)
            sum_list = [i_sim, s_sim, i_s_sim]
        else:
            i_sim = aifsim.get_i_node_sim(g_copy, g1_copy)
            s_sim = aifsim.get_s_node_sim(g_copy, g1_copy)
            i_s_sim = aifsim.get_i_s_node_sim(g_copy, g1_copy)

            i_s_ya_sim = aifsim.get_i_s_ya_node_sim(g_copy, g1_copy)
            l_sim = aifsim.get_l_node_sim(g_copy, g1_copy)
            l_ta_sim = aifsim.get_l_ta_node_sim(g_copy, g1_copy)
            ya_sim = aifsim.get_ya_node_sim(g_copy, g1_copy)
            ta_sim = aifsim.get_ta_node_sim(g_copy, g1_copy)
            l_i_ya_sim = aifsim.get_ya_l_i_node_sim(g_copy, g1_copy)
            l_ta_ya_sim = aifsim.get_l_ta_ya_node_sim(g_copy, g1_copy)
            l_ta_ya_sim = aifsim.get_ya_l_node_sim(g_copy, g1_copy)
            sum_list = [i_sim, s_sim, i_s_sim, i_s_ya_sim, l_sim, l_ta_sim, ya_sim, ta_sim, l_i_ya_sim, l_ta_ya_sim, l_ta_ya_sim]
        sum_tot = sum(sum_list)
        tot = sum_tot/len(sum_list)
        sum_list = np.asarray(sum_list)
        harm = len(sum_list) / np.sum(1.0/sum_list)
        return tot

    @staticmethod
    def rels_to_dict(rels, switched):
        new_list = []
        for rel in rels:

            id_1 = rel[0][0]
            id_2 = rel[1][0]
            text_1 = rel[0][1]
            text_2 = rel[1][1]

            if switched:

                mat_dict = {'ID1': id_2, 'ID2': id_1, 'text1': text_2, 'text2': text_1}
            else:
                mat_dict = {'ID1': id_1, 'ID2': id_2, 'text1': text_1, 'text2': text_2}
            new_list.append(mat_dict)
        return new_list

    @staticmethod
    def get_prop_sim_matrix(graph, graph1):
        centra = Centrality()
        aifsim = Aifsim()


        g_copy = graph.copy()
        g1_copy = graph1.copy()


        g_inodes = centra.get_i_node_list(g_copy)
        g1_inodes = centra.get_i_node_list(g1_copy)
        relsi, valsi, switched = aifsim.text_sim_matrix(g_inodes, g1_inodes)

        #if switched the relations have been switched order so they need reversed when creating the dictionary

        rels_dict = aifsim.rels_to_dict(relsi, switched)

        return rels_dict

    @staticmethod
    def get_loc_sim_matrix(graph, graph1):
        centra = Centrality()
        aifsim = Aifsim()


        g_copy = graph.copy()
        g1_copy = graph1.copy()


        g_lnodes = centra.get_l_node_list(g_copy)
        g1_lnodes = centra.get_l_node_list(g1_copy)
        relsl, valsl, switched = aifsim.text_sim_matrix(g_lnodes, g1_lnodes)

        rels_dict = aifsim.rels_to_dict(relsl, switched)

        return rels_dict

    @staticmethod
    def text_sim_matrix(g_list, g1_list):
        aifsim = Aifsim()
        g_size = len(g_list)
        g1_size = len(g1_list)

        switch_flag = False


        if g_size >= g1_size:
            mat = aifsim.loop_nodes(g_list, g1_list)
            rels, vals = aifsim.select_max_vals(mat, g1_size, g_list, g1_list)
        else:
            switch_flag = True
            mat = aifsim.loop_nodes(g1_list, g_list)
            rels, vals = aifsim.select_max_vals(mat, g_size, g1_list, g_list)

        return rels, vals, switch_flag


    @staticmethod
    def loop_nodes(g_list, g1_list):
        matrix = np.zeros((len(g_list), len(g1_list)))
        for i, node in enumerate(g_list):
            text = node[1]
            text = text.lower()
            for i1, node1 in enumerate(g1_list):

                text1 = node1[1]
                text1 = text1.lower()
                #lev_val = normalized_levenshtein.distance(text, text1)
                lev_val = (fuzz.ratio(text, text1))/100
                matrix[i][i1] = lev_val

        return matrix




    @staticmethod
    def select_max_vals(matrix, smallest_value, g_list, g1_list):
        counter = 0
        lev_vals = []
        lev_rels = []
        index_list = list(range(len(g_list)))
        m_copy = copy.deepcopy(matrix)
        while counter <= smallest_value - 1:
            index_tup = unravel_index(m_copy.argmax(), m_copy.shape)
            #matrix[index_tup[0]][index_tup[1]] = -9999999
            m_copy[index_tup[0]] = 0   # zeroes out row i
            m_copy[:,index_tup[1]] = 0 # zeroes out column i
            lev_rels.append((g_list[index_tup[0]],g1_list[index_tup[1]]))
            lev_vals.append(matrix[index_tup[0]][index_tup[1]])
            index_list.remove(index_tup[0])
            counter = counter + 1
        for vals in index_list:
            lev_rels.append((g_list[vals],''))
            lev_vals.append(0)
        return lev_rels, lev_vals

    @staticmethod
    def convert_to_dict(conf_matrix):
        values = []
        dicts = {}
        for i, col in enumerate(conf_matrix):
            dicts[i] = {}
            for j, row in enumerate(col):
                dicts[i][j] = row
        return dicts


    @staticmethod
    def get_mean_of_list(a):
        val_tot = sum(a)
        tot = val_tot/len(a)
        return tot


    @staticmethod
    def get_l_i_mean(l, i):
        return (l+i)/2


    @staticmethod
    def get_graph_sim(aif_id1, aif_id2):
        centra = Centrality()
        aifsim = Aifsim()
        graph, json = aifsim.get_graph(aif_id1, centra)
        graph1, json1 = aifsim.get_graph(aif_id2, centra)
        graph = centra.remove_iso_analyst_nodes(graph)
        graph1 = centra.remove_iso_analyst_nodes(graph1)
        rep_form = aifsim.is_iat(graph, graph1, centra)
        g_copy = graph.copy()
        g1_copy = graph1.copy()
        graph_mean = 0
        text_mean = 0
        overall_mean = 0
        if rep_form == 'diff':
            return 'Error'
        else:
            graph_mean = aifsim.call_diagram_parts_and_sum(g_copy, g1_copy, rep_form)
        if rep_form == 'aif':
            g_inodes = centra.get_i_node_list(g_copy)
            g1_inodes = centra.get_i_node_list(g1_copy)
            relsi, valsi = aifsim.text_sim_matrix(g_inodes, g1_inodes)
            i_mean = aifsim.get_mean_of_list(valsi)
            text_mean = i_mean
        else:
            g_inodes = centra.get_i_node_list(g_copy)
            g1_inodes = centra.get_i_node_list(g1_copy)
            g_lnodes = centra.get_l_node_list(g_copy)
            g1_lnodes = centra.get_l_node_list(g1_copy)
            relsi, valsi = aifsim.text_sim_matrix(g_inodes, g1_inodes)
            relsl, valsl = aifsim.text_sim_matrix(g_lnodes, g1_lnodes)
            i_mean = aifsim.get_mean_of_list(valsi)
            l_mean = aifsim.get_mean_of_list(valsl)
            text_mean = aifsim.get_l_i_mean(l_mean, i_mean)

        overall_score = aifsim.get_l_i_mean(text_mean, graph_mean)
        return overall_score, text_mean, graph_mean


    #Get TA Anchor RA, CA, MA - Requires CA_anchor ma_anchor and ra_anchor then combination of confusion matrices

    @staticmethod
    def ra_anchor(graph1, graph2):

        conf_matrix = [[0, 0],[0, 0]]

        cent = Centrality()
        ras1 = cent.get_ras(graph1)
        ras2 = cent.get_ras(graph2)

        ra1_len = len(ras1)
        ra2_len = len(ras2)

        if ra1_len > 0 and ra2_len > 0:
            if ra1_len > ra2_len:
                for ra_i, ra in enumerate(ras1):
                        ras2_id = ''
                        yas1 = get_ya_nodes_from_prop(ra, graph1)
                        try:
                            ras2_id = ras2[ra_i]
                        except:
                            ras2_id = ''

                        if ras2_id == '':
                            #conf_matrix[index][len(all_ya_text) + 1] =  conf_matrix[index][len(all_ya_text) + 1] + 1
                            conf_matrix[1][0] =  conf_matrix[1][0] + 1
                        else:
                            yas2 = get_ya_nodes(ras2_id, graph2)
                            if yas1 == yas2:

                                conf_matrix[0][0] =  conf_matrix[0][0] + 1
                            else:
                                conf_matrix[1][0] =  conf_matrix[1][0] + 1

            elif ra2_len > ra1_len:
                for ra_i, ra in enumerate(ras2):
                        ras1_id = ''
                        yas2 = get_ya_nodes_from_prop(ra, graph2)
                        try:
                            ras1_id = ras1[ra_i]
                        except:
                            ras1_id = ''

                        if ras1_id == '':
                            #conf_matrix[index][len(all_ya_text) + 1] =  conf_matrix[index][len(all_ya_text) + 1] + 1
                            conf_matrix[0][1] =  conf_matrix[0][1] + 1
                        else:
                            yas1 = get_ya_nodes(ras1_id, graph1)
                            if yas1 == yas2:

                                conf_matrix[0][0] =  conf_matrix[0][0] + 1
                            else:
                                conf_matrix[0][1] =  conf_matrix[0][1] + 1

            else:
                for ra_i, ra in enumerate(ras1):
                    ya1 = get_ya_nodes_from_prop(ra, graph1)
                    ya2 = get_ya_nodes_from_prop(ras2[ra_i], graph2)

                    if ya1 == ya2:
                        conf_matrix[0][0] =  conf_matrix[0][0] + 1
                    else:
                        conf_matrix[1][0] =  conf_matrix[1][0] + 1

        elif ra1_len == 0 and ra2_len == 0:
            conf_matrix[1][1] =  conf_matrix[1][1] + 1

        elif ra1_len == 0:
            conf_matrix[0][1] =  conf_matrix[0][1] + ra2_len
        elif ra2_len == 0:
            conf_matrix[1][0] =  conf_matrix[1][0] + ra1_len

        return conf_matrix

    @staticmethod
    def ma_anchor(graph1, graph2):

        conf_matrix = [[0, 0],[0, 0]]

        cent = Centrality()
        cas1 = cent.get_mas(graph1)
        cas2 = cent.get_mas(graph2)

        ca1_len = len(cas1)
        ca2_len = len(cas2)

        if ca1_len > 0 and ca2_len > 0:
            if ca1_len > ca2_len:
                for ca_i, ca in enumerate(cas1):
                        cas2_id = ''
                        yas1 = get_ya_nodes_from_prop(ca, graph1)
                        try:
                            cas2_id = cas2[ca_i]
                        except:
                            cas2_id = ''

                        if cas2_id == '':
                            #conf_matrix[index][len(all_ya_text) + 1] =  conf_matrix[index][len(all_ya_text) + 1] + 1
                            conf_matrix[1][0] =  conf_matrix[1][0] + 1
                        else:
                            yas2 = get_ya_nodes(cas2_id, graph2)
                            if yas1 == yas2:

                                conf_matrix[0][0] =  conf_matrix[0][0] + 1
                            else:
                                conf_matrix[1][0] =  conf_matrix[1][0] + 1

            elif ca2_len > ca1_len:
                for ca_i, ca in enumerate(cas2):
                        cas1_id = ''
                        yas2 = get_ya_nodes_from_prop(ca, graph2)
                        try:
                            cas1_id = cas1[ca_i]
                        except:
                            cas1_id = ''

                        if cas1_id == '':
                            #conf_matrix[index][len(all_ya_text) + 1] =  conf_matrix[index][len(all_ya_text) + 1] + 1
                            conf_matrix[0][1] =  conf_matrix[0][1] + 1
                        else:
                            yas1 = get_ya_nodes(cas1_id, graph1)
                            if yas1 == yas2:

                                conf_matrix[0][0] =  conf_matrix[0][0] + 1
                            else:
                                conf_matrix[0][1] =  conf_matrix[0][1] + 1

            else:
                for ca_i, ca in enumerate(cas1):
                    ya1 = get_ya_nodes_from_prop(ca, graph1)
                    ya2 = get_ya_nodes_from_prop(cas2[ca_i], graph2)

                    if ya1 == ya2:
                        conf_matrix[0][0] =  conf_matrix[0][0] + 1
                    else:
                        conf_matrix[1][0] =  conf_matrix[1][0] + 1

        elif ca1_len == 0 and ca2_len == 0:
            conf_matrix[1][1] =  conf_matrix[1][1] + 1

        elif ca1_len == 0:
            conf_matrix[0][1] =  conf_matrix[0][1] + ca2_len
        elif ca2_len == 0:
            conf_matrix[1][0] =  conf_matrix[1][0] + ca1_len

        return conf_matrix

    @staticmethod
    def ca_anchor(graph1, graph2):

        conf_matrix = [[0, 0],[0, 0]]

        cent = Centrality()
        cas1 = cent.get_cas(graph1)
        cas2 = cent.get_cas(graph2)

        ca1_len = len(cas1)
        ca2_len = len(cas2)

        if ca1_len > 0 and ca2_len > 0:
            if ca1_len > ca2_len:
                for ca_i, ca in enumerate(cas1):
                        cas2_id = ''
                        yas1 = get_ya_nodes_from_prop(ca, graph1)
                        try:
                            cas2_id = cas2[ca_i]
                        except:
                            cas2_id = ''

                        if cas2_id == '':
                            #conf_matrix[index][len(all_ya_text) + 1] =  conf_matrix[index][len(all_ya_text) + 1] + 1
                            conf_matrix[1][0] =  conf_matrix[1][0] + 1
                        else:
                            yas2 = get_ya_nodes(cas2_id, graph2)
                            if yas1 == yas2:

                                conf_matrix[0][0] =  conf_matrix[0][0] + 1
                            else:
                                conf_matrix[1][0] =  conf_matrix[1][0] + 1

            elif ca2_len > ca1_len:
                for ca_i, ca in enumerate(cas2):
                        cas1_id = ''
                        yas2 = get_ya_nodes_from_prop(ca, graph2)
                        try:
                            cas1_id = cas1[ca_i]
                        except:
                            cas1_id = ''

                        if cas1_id == '':
                            #conf_matrix[index][len(all_ya_text) + 1] =  conf_matrix[index][len(all_ya_text) + 1] + 1
                            conf_matrix[0][1] =  conf_matrix[0][1] + 1
                        else:
                            yas1 = get_ya_nodes(cas1_id, graph1)
                            if yas1 == yas2:

                                conf_matrix[0][0] =  conf_matrix[0][0] + 1
                            else:
                                conf_matrix[0][1] =  conf_matrix[0][1] + 1

            else:
                for ca_i, ca in enumerate(cas1):
                    ya1 = get_ya_nodes_from_prop(ca, graph1)
                    ya2 = get_ya_nodes_from_prop(cas2[ca_i], graph2)

                    if ya1 == ya2:
                        conf_matrix[0][0] =  conf_matrix[0][0] + 1
                    else:
                        conf_matrix[1][0] =  conf_matrix[1][0] + 1

        elif ca1_len == 0 and ca2_len == 0:
            conf_matrix[1][1] =  conf_matrix[1][1] + 1

        elif ca1_len == 0:
            conf_matrix[0][1] =  conf_matrix[0][1] + ca2_len
        elif ca2_len == 0:
            conf_matrix[1][0] =  conf_matrix[1][0] + ca1_len

        return conf_matrix


