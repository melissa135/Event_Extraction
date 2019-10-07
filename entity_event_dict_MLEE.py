entity_dict = { 'Organism':1, 'Organism_subdivision':2, 'Anatomical_system':3, 'Organ':4,
                'Multi':5, 'Tissue':6, 'Cell':7, 'Cellular_component':8, 'Developing_anatomical_structure':9,
                'Organism_substance':10, 'Immaterial_anatomical_entity':11, 'Pathological_formation':12, 
                'Drug_or_compound':13, 'Gene_or_gene_product':14 }

event_dict = { 'Development':1, 'Blood_vessel_development':2, 'Growth':3, 'Death':4,
               'Breakdown':5, 'Cell_proliferation':6, 'Remodeling':7, 'Reproduction':8,
               'Synthesis':9, 'Catabolism':10, 'Gene_expression':11, 'Metabolism':12,
               'Transcription':13, 'Phosphorylation':14, 'Pathway':15,
               'Binding':16, 'Localization':17, 'Regulation':18,
               'Positive_regulation':19, 'Negative_regulation':20, 'Planned_process':21,
               'Dephosphorylation':22 }

e_dict = { 'Organism':1, 'Organism_subdivision':2, 'Anatomical_system':3, 'Organ':4,
           'Multi':5, 'Tissue':6, 'Cell':7, 'Cellular_component':8, 'Developing_anatomical_structure':9,
           'Organism_substance':10, 'Immaterial_anatomical_entity':11, 'Pathological_formation':12, 'Drug_or_compound':13,
           'Gene_or_gene_product':14,
           'Development':15, 'Blood_vessel_development':16, 'Growth':17, 'Death':18,
           'Breakdown':19, 'Cell_proliferation':20, 'Remodeling':21, 'Reproduction':22,
           'Synthesis':23, 'Catabolism':24, 'Gene_expression':25, 'Metabolism':26,
           'Transcription':27, 'Phosphorylation':28, 'Pathway':29,
           'Binding':30, 'Localization':31, 'Regulation':32,
           'Positive_regulation':33, 'Negative_regulation':34, 'Planned_process':35,
           'Dephosphorylation':36,
           'OTHER':37 }

relation_dict = { 'Theme':0, 'Theme2':1, 'AtLoc':2, 'Site':3, 'ToLoc':4, 'Participant':5,
                  'Participant2':6, 'FromLoc':7, 'Cause':8, 'Instrument':9, 'Instrument2':10 }
# think about multi-tissue, which take '-' inside

valid_para_types_dict = { 'Acetylation':{'Theme'},
                          'Binding':{'Theme','Theme2','Site'},
                          'Blood_vessel_development':{'Theme','AtLoc'},
                          'Breakdown':{'Theme'},
                          'Catabolism':{'Theme'},
                          'Cell_division':{'Theme'},
                          'Cell_proliferation':{'Theme'},
                          'DNA_methylation':{'Theme','Site'},
                          'Death':{'Theme'},
                          'Dephosphorylation':{'Theme','Site'},
                          'Development':{'Theme'},
                          'Dissociation':{'Theme'},
                          'Gene_expression':{'Theme'},
                          'Growth':{'Theme'},
                          'Localization':{'Theme','AtLoc','FromLoc','ToLoc'},
                          'Metabolism':{'Theme'},
                          'Negative_regulation':{'Theme','Cause','Site'},
                          'Pathway':{'Theme','Participant','Participant2'},
                          'Phosphorylation':{'Theme','Site'},
                          'Planned_process':{'Theme','Instrument','Instrument2'},
                          'Positive_regulation':{'Theme','Site','Cause'},
                          'Protein_processing':{'Theme'},
                          'Regulation':{'Theme','Cause','Site'},
                          'Remodeling':{'Theme'},
                          'Reproduction':{'Theme'},
                          'Synthesis':{'Theme'},
                          'Transcription':{'Theme'},
                          'Translation':{'Theme'},
                          'Ubiquitination':{'Theme'} }

paras_with_none_dict = { 'Acetylation':{},
                         'Binding':{'Theme2','Site'}, #'Theme2','Site'
                         'Blood_vessel_development':{'Theme','AtLoc'},
                         'Breakdown':{},
                         'Catabolism':{},
                         'Cell_division':{},
                         'Cell_proliferation':{},
                         'DNA_methylation':{'Site'},
                         'Death':{},
                         'Dephosphorylation':{'Site'},
                         'Development':{},
                         'Dissociation':{},
                         'Gene_expression':{},
                         'Growth':{},
                         'Localization':{'AtLoc','FromLoc','ToLoc'},
                         'Metabolism':{},
                         'Negative_regulation':{'Cause','Site'},
                         'Pathway':{'Theme','Participant','Participant2'},
                         'Phosphorylation':{'Site'},
                         'Planned_process':{'Theme','Instrument2','Instrument'}, # ,'Instrument'
                         'Positive_regulation':{'Site','Cause'},
                         'Protein_processing':{},
                         'Regulation':{'Cause','Site'},
                         'Remodeling':{},
                         'Reproduction':{},
                         'Synthesis':{},
                         'Transcription':{},
                         'Translation':{},
                         'Ubiquitination':{} }
