entity_dict = { 'Organism':1, 'Organism_subdivision':2, 'Anatomical_system':3, 'Organ':4,
                'Multi':5, 'Tissue':6, 'Developing_anatomical_structure':7, 'Cell':8, 'Cellular_component':9,
                'Organism_substance':10, 'Immaterial_anatomical_entity':11, 'Pathological_formation':12, 'Cancer':13, 'Gene_or_gene_product':14,
                'Protein_domain_or_region':15, 'DNA_domain_or_region':16, 'Simple_chemical':17, 'Amino_acid':18 }

event_dict = { 'Development':1, 'Blood_vessel_development':2, 'Growth':3, 'Death':4,
               'Cell_death':5, 'Breakdown':6, 'Cell_proliferation':7, 'Cell_division':8, 'Cell_differentiation':9,
               'Remodeling':10, 'Reproduction':11, 'Mutation':12, 'Carcinogenesis':13, 'Cell_transformation':14,
               'Metastasis':15, 'Infection':16, 'Metabolism':17, 'Synthesis':18, 'Catabolism':19,
               'Amino_acid_catabolism':20, 'Glycolysis':21, 'Gene_expression':22, 'Transcription':23, 'Translation':24,
               'Protein_processing':25, 'Phosphorylation':26, 'Pathway':27, 'Binding':28, 'Dissociation':29,
               'Localization':30, 'Regulation':31, 'Positive_regulation':32, 'Negative_regulation':33, 'Planned_process':34,
               'Ubiquitination':35, 'Dephosphorylation':36, 'DNA_demethylation':37, 'Acetylation':38, 'DNA_methylation':39,
               'Glycosylation':40 }

e_dict = { 'None':0, 'Development':1, 'Blood_vessel_development':2, 'Growth':3, 'Death':4,
           'Cell_death':5, 'Breakdown':6, 'Cell_proliferation':7, 'Cell_division':8, 'Cell_differentiation':9,
           'Remodeling':10, 'Reproduction':11, 'Mutation':12, 'Carcinogenesis':13, 'Cell_transformation':14,
           'Metastasis':15, 'Infection':16, 'Metabolism':17, 'Synthesis':18, 'Catabolism':19,
           'Amino_acid_catabolism':20, 'Glycolysis':21, 'Gene_expression':22, 'Transcription':23, 'Translation':24,
           'Protein_processing':25, 'Phosphorylation':26, 'Pathway':27, 'Binding':28, 'Dissociation':29,
           'Localization':30, 'Regulation':31, 'Positive_regulation':32, 'Negative_regulation':33, 'Planned_process':34,
           'Ubiquitination':35, 'Dephosphorylation':36, 'DNA_demethylation':37, 'Acetylation':38, 'DNA_methylation':39,
           'Glycosylation':40, 'Organism':41, 'Organism_subdivision':42, 'Anatomical_system':43, 'Organ':44,
           'Multi':45, 'Tissue':46, 'Developing_anatomical_structure':47, 'Cell':48, 'Cellular_component':49,
           'Organism_substance':50, 'Immaterial_anatomical_entity':51, 'Pathological_formation':52, 'Cancer':53, 'Gene_or_gene_product':54,
           'Protein_domain_or_region':55, 'DNA_domain_or_region':56, 'Simple_chemical':57, 'Amino_acid':58, \
           'OTHER':59 }

relation_dict = { 'Theme':0, 'Theme2':1, 'AtLoc':2, 'Site':3, 'ToLoc':4, 'Participant':5,
                  'Participant2':6, 'FromLoc':7, 'Cause':8, 'Instrument':9, 'Instrument2':10 }
# think about multi-tissue, which take '-' inside

valid_para_types_dict = { 'Development':{'Theme'},
                          'Growth':{'Theme'},
                          'Death':{'Theme'},
                          'Breakdown':{'Theme'},
                          'Cell_proliferation':{'Theme'},
                          'Cell_division':{'Theme'},
                          'Remodeling':{'Theme'},
                          'Reproduction':{'Theme'},
                          'Metabolism':{'Theme'},
                          'Synthesis':{'Theme'},
                          'Catabolism':{'Theme'},
                          'Transcription':{'Theme'},
                          'Translation':{'Theme'},
                          'Protein_processing':{'Theme'},
                          'Cell_death':{'Theme'},
                          'Amino_acid_catabolism':{'Theme'},
                          'Glycolysis':{'Theme'},
                          'Cell_differentiation':{'Theme','AtLoc'},
                          'Cell_transformation':{'Theme','AtLoc'},
                          'Blood_vessel_development':{'Theme','AtLoc'},
                          'Carcinogenesis':{'Theme','AtLoc'},
                          'Mutation':{'Theme','AtLoc','Site'},
                          'Metastasis':{'Theme','ToLoc'},
                          'Infection':{'Theme','Participant'},
                          'Gene_expression':{'Theme'}, # ,'Theme2'
                          'Phosphorylation':{'Theme','Site'},
                          'Ubiquitination':{'Theme'}, # ,'Site'
                          'Dephosphorylation':{'Theme','Site'},
                          'DNA_demethylation':{'Theme','Site'},
                          'Acetylation':{'Theme','Site'},
                          'DNA_methylation':{'Theme','Site'},
                          'Glycosylation':{'Theme'}, #,'Site'
                          'Dissociation':{'Theme','Theme2'}, # site -> theme2
                          'Pathway':{'Theme','Participant','Participant2'},
                          'Binding':{'Theme','Theme2','Site'},
                          'Localization':{'Theme','Theme2','AtLoc','FromLoc','ToLoc'},
                          'Regulation':{'Theme','Cause'},
                          'Negative_regulation':{'Theme','Cause'},
                          'Positive_regulation':{'Theme','Cause'},
                          'Planned_process':{'Theme','Theme2','Instrument','Instrument2'} }

paras_with_none_dict = { 'Development':{},
                         'Growth':{},
                         'Death':{},
                         'Breakdown':{},
                         'Cell_proliferation':{},
                         'Cell_division':{},
                         'Remodeling':{},
                         'Reproduction':{},
                         'Metabolism':{},
                         'Synthesis':{},
                         'Catabolism':{},
                         'Transcription':{},
                         'Translation':{},
                         'Protein_processing':{},
                         'Cell_death':{'Theme'},
                         'Amino_acid_catabolism':{'Theme'},
                         'Glycolysis':{'Theme'},
                         'Cell_differentiation':{'Theme','AtLoc'}, # -'Theme'
                         'Cell_transformation':{'Theme','AtLoc'},
                         'Blood_vessel_development':{'Theme','AtLoc'},
                         'Carcinogenesis':{'Theme','AtLoc'},
                         'Mutation':{'Theme','AtLoc','Site'},
                         'Metastasis':{'Theme','ToLoc'},
                         'Infection':{'Theme','Participant'},
                         'Gene_expression':{}, # 'Theme2'
                         'Phosphorylation':{'Site'},
                         'Ubiquitination':{}, # 'Site'
                         'Dephosphorylation':{'Site'},
                         'DNA_demethylation':{'Site'},
                         'Acetylation':{'Site'},
                         'DNA_methylation':{'Site'},
                         'Glycosylation':{}, #'Site'
                         'Dissociation':{'Theme2'},# site -> theme2
                         'Pathway':{'Theme','Participant','Participant2'},
                         'Binding':{'Theme2','Site'},
                         'Localization':{'Theme2','AtLoc','FromLoc','ToLoc'},
                         'Regulation':{'Cause'},
                         'Negative_regulation':{'Cause'},
                         'Positive_regulation':{'Cause'},
                         'Planned_process':{'Theme','Theme2','Instrument','Instrument2'} }
