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
           'Protein_domain_or_region':55, 'DNA_domain_or_region':56, 'Simple_chemical':57, 'Amino_acid':58,\
           'OTHER':59 }

relation_dict = { 'Theme':0, 'Theme2':1, 'AtLoc':2, 'Site':3, 'ToLoc':4, 'Participant':5,
                  'Participant2':6, 'FromLoc':7, 'Cause':8, 'Instrument':9, 'Instrument2':10 }
# think about multi-tissue, which take '-' inside
