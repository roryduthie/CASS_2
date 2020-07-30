from application.centrality import Centrality
import numpy as np
from numpy import unravel_index
import copy
import gmatch4py as gm
import networkx as nx
from fuzzywuzzy import fuzz

class Aifsim:
    @staticmethod
    def get_graph(aif_id, cent):
        dir_path = 'http://www.aifdb.org/json/' + str(aif_id)
        graph, json = cent.get_graph_url(dir_path)
        return graph, json
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
    def text_sim_matrix(g_list, g1_list):
        aifsim = Aifsim()
        g_size = len(g_list)
        g1_size = len(g1_list)


        if g_size >= g1_size:
            mat = aifsim.loop_nodes(g_list, g1_list)
            rels, vals = aifsim.select_max_vals(mat, g1_size, g_list, g1_list)
        else:
            mat = aifsim.loop_nodes(g1_list, g_list)
            rels, vals = aifsim.select_max_vals(mat, g_size, g1_list, g_list)

        return rels, vals


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


