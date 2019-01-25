from entity_event_dict_CG import *


class Event :
    def __init__(self, event_id, paras, support, modification):
        self.event_id = event_id
        self.paras = paras
        self.support = support
        self.modification = modification

    def get_modification(self):
        if self.modification == 0 :
            return 'None'
        elif self.modification == 1 :
            return 'Speculation'
        elif self.modification == 2 :
            return 'Negation'


def remove_subset(valid_combination):

    valid_combination_without_none = [] # remove none-parameter
    for v_c in valid_combination :
        v_c_without_none = set()
        for para in v_c.paras :
            if not para[1] == 'None' :
                v_c_without_none.add(para)
        event = Event(v_c.event_id,v_c_without_none,v_c.support,v_c.modification)
        valid_combination_without_none.append(event)

    valid_combination = []
    for v_c in valid_combination_without_none :
        subset_flag = False
        for v_c_ in valid_combination_without_none :
            if v_c == v_c_ :
                continue
            if v_c.paras.issubset(v_c_.paras) or v_c_.paras.issubset(v_c.paras) :
                if v_c.support < v_c_.support : # delete the small-support one
                    subset_flag = True
        if not subset_flag :
            valid_combination.append(v_c) # remove subset

    return valid_combination


def complete_combination(para_dict):
    # para_dict: dict of para_type -> list of para_key
    key = para_dict.keys()[0]
    if len(para_dict.keys()) == 1 :
        complete_list = [ [item] for item in para_dict[key] ] # list of lists
        return [key],complete_list

    sub_para_dict = para_dict.copy()
    sub_para_dict.pop(key)
    keys,sub_list = complete_combination(sub_para_dict)
    keys.append(key)
    complete_list = []
    for sub_set in sub_list :
        for new_item in para_dict[key]:
            tmp_set = sub_set[:]
            if (new_item != 'None')and(new_item == tmp_set[-1]) :
                continue
            tmp_set.append(new_item)
            complete_list.append(tmp_set)
    # complete_list: list of all valid combinations (list, in keys order)
    return keys,complete_list


def generate_all_combination(e_type,all_paras,e_loc):
    # etype = src_type[src_key]
    # all_paras = event_para_[src_key]
    valid_para_types = valid_para_types_dict[e_type]
    paras_with_none = paras_with_none_dict[e_type]
    valid_paras = dict()

    for vpt in valid_para_types :
        valid_paras[vpt] = []
    for para in all_paras :
        p_type, p_key = para
        if p_type in valid_para_types :
            valid_paras[p_type].append(p_key)

    if e_type == 'Binding':
        valid_paras['Theme2'].extend(valid_paras['Theme'])
    if e_type == 'Pathway':
        valid_paras['Participant2'].extend(valid_paras['Participant'])

    for pwn in paras_with_none :
        valid_paras[pwn].append('None')

    ks,c_c = complete_combination(valid_paras)

    range_lists = []
    para_sets = []
    for c in c_c:
        range_list = [ (ks[j],e_loc[c[j]]) for j in range(0,len(ks)) ]
        para_set = [ (ks[j],c[j]) for j in range(0,len(ks)) ]
        range_lists.append(range_list)
        para_sets.append(para_set)

    return range_lists,para_sets
