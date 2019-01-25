entity_dict = { 'Simple_chemical':1, 'Gene_or_gene_product':2, 'Complex':3, 'Cellular_component':4 }

event_dict = { 'Conversion':1, 'Phosphorylation':2, 'Dephosphorylation':3, 'Acetylation':4,
               'Deacetylation':5, 'Methylation':6, 'Demethylation':7, 'Ubiquitination':8,
               'Deubiquitination':9, 'Localization':10, 'Transport':11, 'Gene_expression':12,
               'Transcription':13, 'Translation':14, 'Degradation':15, 'Activation':16,
               'Inactivation':17, 'Binding':18, 'Dissociation':19, 'Regulation':20, 
               'Positive_regulation':21, 'Negative_regulation':22, 'Pathway':23 }

e_dict = { 'None':0, 'Conversion':1, 'Phosphorylation':2, 'Dephosphorylation':3, 'Acetylation':4,
           'Deacetylation':5, 'Methylation':6, 'Demethylation':7, 'Ubiquitination':8,
           'Deubiquitination':9, 'Localization':10, 'Transport':11, 'Gene_expression':12,
           'Transcription':13, 'Translation':14, 'Degradation':15, 'Activation':16,
           'Inactivation':17, 'Binding':18, 'Dissociation':19, 'Regulation':20, 
           'Positive_regulation':21, 'Negative_regulation':22, 'Pathway':23,
           'Simple_chemical':24, 'Gene_or_gene_product':25, 'Complex':26, 'Cellular_component':27,
           'OTHER':28 }
# think about multi-tissue, which take '-' inside

valid_para_types_dict = { 'Conversion':{'Theme','Product'}, # ,'Theme2'
                          'Phosphorylation':{'Theme','Cause','Site'},
                          'Dephosphorylation':{'Theme','Cause','Site'},
                          'Acetylation':{'Theme','Cause','Site'},
                          'Deacetylation':{'Theme'},
                          'Methylation':{'Theme','Cause','Site'},
                          'Demethylation':{'Theme','Cause','Site'},
                          'Ubiquitination':{'Theme','Cause','Site'},
                          'Deubiquitination':{'Theme'},
                          'Localization':{'Theme','AtLoc','ToLoc'}, # ,'FromLoc'
                          'Transport':{'Theme','FromLoc','ToLoc'},
                          'Gene_expression':{'Theme'},
                          'Transcription':{'Theme'},
                          'Translation':{'Theme'},
                          'Degradation':{'Theme'},
                          'Activation':{'Theme','Cause'},
                          'Inactivation':{'Theme','Cause'},
                          'Binding':{'Theme','Theme2'}, #,'Product'
                          'Dissociation':{'Theme','Product','Product2'},
                          'Regulation':{'Theme','Cause'}, 
                          'Positive_regulation':{'Theme','Cause'},
                          'Negative_regulation':{'Theme','Cause'},
                          'Pathway':{'Participant','Participant2'} }

paras_with_none_dict = { 'Conversion':{'Theme','Product'}, #,'Theme2'
                         'Phosphorylation':{'Cause','Site'},
                         'Dephosphorylation':{'Cause','Site'},
                         'Acetylation':{'Cause','Site'},
                         'Deacetylation':{},
                         'Methylation':{'Cause','Site'},
                         'Demethylation':{'Cause','Site'},
                         'Ubiquitination':{'Cause','Site'},
                         'Deubiquitination':{},
                         'Localization':{'AtLoc','ToLoc'}, # ,'FromLoc'
                         'Transport':{'FromLoc','ToLoc'},
                         'Gene_expression':{},
                         'Transcription':{},
                         'Translation':{},
                         'Degradation':{},
                         'Activation':{'Cause'},
                         'Inactivation':{'Cause'},
                         'Binding':{'Theme2'}, # {'Theme','Theme2','Product'}
                         'Dissociation':{'Theme','Product','Product2'},
                         'Regulation':{'Cause'}, 
                         'Positive_regulation':{'Cause'},
                         'Negative_regulation':{'Cause'},
                         'Pathway':{'Participant','Participant2'} }
