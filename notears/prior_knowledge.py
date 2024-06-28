'''
LUCAS = {
    0: "Smoking",
    1: "Yellow_Fingers",
    2: "Anxiety",
    3: "Peer_Pressure",
    4: "Genetics",
    5: "Attention_Disorder",
    6: "Born_an_Even_Day",
    7: "Car_Accident",
    8: "Fatigue",
    9: "Allergy",
    10: "Coughing",
    11: "Lung_Cancer"
}

Asia = {
    0: 'VisitAsia', 
    1: 'TB', 
    2: 'Smoking', 
    3: 'Cancer', 
    4: 'TBorCancer', 
    5: 'XRay', 
    6: 'Bronchitis', 
    7: 'Dysphenia'
}

SACHS = {
    0: 'praf',
    1: 'pmek',
    2: 'plcg',
    3: 'PIP2',
    4: 'PIP3',
    5: 'p44/42',
    6: 'pakts473',
    7: 'PKA',
    8: 'PKC',
    9: 'P38',
    10: 'pjnk'
}

'''

import numpy as np


class PriorKnowledge:
    def __init__(self, dataset, true_graph = False, LLMs = ['GPT3', 'GPT4', 'Gemini']):
        self.prior_knowledge = {}
        if dataset not in ['LUCAS', 'Asia', 'SACHS', 'Survey', 'Earthquake', 'Child', 'Alarm']:
            raise ValueError("The dataset should be one of the following: ['LUCAS', 'Asia', 'SACHS', 'Survey', 'Earthquake', 'Child', 'Alarm']")
        
        self.LLMs = LLMs
        if true_graph:
            print("Using true graph as prior knowledge")
            if dataset == 'LUCAS':
                self.add_LUCAS_true_knowledge()
            if dataset == 'Asia':
                self.add_Asia_true_knowledge()
            if dataset == 'SACHS':
                self.add_SACHS_true_knowledge()
            if dataset == 'Survey':
                self.add_Survey_true_knowledge()
        else:
            if dataset == 'LUCAS':
                self.add_LUCAS_knowledge()
            if dataset == 'Asia':
                self.add_Asia_knowledge()
            if dataset == 'SACHS':
                self.add_SACHS_knowledge()
            if dataset == 'Survey':
                self.add_Survey_knowledge()
            if dataset == 'Earthquake':
                self.add_Earthquake_knowledge()
            if dataset == 'Child':
                self.add_Child_knowledge()
            if dataset == 'Alarm':
                self.add_Alarm_knowledge()

        self.dataset = dataset
        self.intersection_result = {}
        self.LLM_weights = {}
        self.add_intersection_result()

    def calculate_LLMs_weight(self, X):
        for dataset in self.prior_knowledge:
            if dataset == self.dataset:
                for model in self.prior_knowledge[dataset]:
                    if model not in self.LLM_weights:
                        self.LLM_weights[model] = {}
                    LLM_result = self.prior_knowledge[dataset][model]
                    LLM_result[LLM_result == 2] = 0

                    M = X @ LLM_result
                    R = X - M
                    self.LLM_weights[model] = 0.5 / X.shape[0] * (R ** 2).sum()
        
        self._transfer_LLM_score_to_weights()
        print(self.LLM_weights)

    def _transfer_LLM_score_to_weights(self):

        if len(self.LLM_weights) == 0:
            return
        
        if len(self.LLM_weights) == 1:
            self.LLM_weights[list(self.LLM_weights.keys())[0]] = 1
            return

        result = np.array([])
        for model in self.LLM_weights:
            result = np.append(result, -self.LLM_weights[model])

        z_score = (result - np.mean(result)) / np.std(result)
        softmax_result = np.exp(z_score) / np.sum(np.exp(z_score))

        for i, model in enumerate(self.LLM_weights):
            self.LLM_weights[model] = softmax_result[i] 

    def get_prior_knowledge(self):
        return self.prior_knowledge

    def get_dataset(self):
        return self.dataset



    def add_intersection_result(self):
        for dataset in self.prior_knowledge:
            p = 0
            edge_sets = []
            for method in self.prior_knowledge[dataset]:
                edge_sets.append(set(zip(*np.where(self.prior_knowledge[dataset][method] == 1))))
                p = self.prior_knowledge[dataset][method].shape[0]

            re = edge_sets[0].intersection(*edge_sets[1:])
            result_matrix = np.zeros((p, p))
            for edge in re:
                result_matrix[edge] = 1
            self.intersection_result[dataset] = result_matrix
        


    def add_Alarm_knowledge(self):
        items = [
                    "LVFAILURE", "HISTORY", "LVEDVOLUME", "CVP", "PCWP", "HYPOVOLEMIA",
                    "STROKEVOLUME", "ERRLOWOUTPUT", "HRBP", "HR", "ERRCAUTER", "HREKG",
                    "HRSAT", "ANAPHYLAXIS", "TPR", "ARTCO2", "EXPCO2", "VENTLUNG", "INTUBATION",
                    "MINVOL", "FIO2", "PVSAT", "VENTALV", "SAO2", "SHUNT", "PULMEMBOLUS", "PAP",
                    "PRESS", "KINKEDTUBE", "VENTTUBE", "MINVOLSET", "VENTMACH", "DISCONNECT",
                    "CATECHOL", "INSUFFANESTH", "CO", "BP"
                ]
        alarm_variables = {item: index for index, item in enumerate(items)}

        if 'GPT3' in self.LLMs:
            # GPT3 knowledge
            self.prior_knowledge['Alarm'] = {}
            self.prior_knowledge['Alarm']['GPT3'] = np.ones((37, 37)) * 2
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['LVFAILURE']][alarm_variables['STROKEVOLUME']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['LVFAILURE']][alarm_variables['ERRLOWOUTPUT']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['LVFAILURE']][alarm_variables['HRBP']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['HISTORY']][alarm_variables['LVFAILURE']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['LVEDVOLUME']][alarm_variables['LVFAILURE']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['CVP']][alarm_variables['LVFAILURE']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['PCWP']][alarm_variables['LVFAILURE']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['HYPOVOLEMIA']][alarm_variables['LVFAILURE']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['STROKEVOLUME']][alarm_variables['ERRLOWOUTPUT']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['STROKEVOLUME']][alarm_variables['LVFAILURE']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['HRBP']][alarm_variables['LVFAILURE']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['ANAPHYLAXIS']][alarm_variables['ERRLOWOUTPUT']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['TPR']][alarm_variables['LVFAILURE']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['ARTCO2']][alarm_variables['VENTLUNG']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['EXPCO2']][alarm_variables['VENTLUNG']] = 1      
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['INTUBATION']][alarm_variables['VENTLUNG']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['MINVOL']][alarm_variables['VENTLUNG']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['FIO2']][alarm_variables['VENTLUNG']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['PVSAT']][alarm_variables['VENTALV']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['SAO2']][alarm_variables['VENTALV']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['SHUNT']][alarm_variables['VENTALV']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['PULMEMBOLUS']][alarm_variables['PAP']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['PRESS']][alarm_variables['KINKEDTUBE']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['VENTTUBE']][alarm_variables['KINKEDTUBE']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['MINVOLSET']][alarm_variables['VENTMACH']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['VENTMACH']][alarm_variables['DISCONNECT']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['CATECHOL']][alarm_variables['BP']] = 1
            self.prior_knowledge['Alarm']['GPT3'][alarm_variables['INSUFFANESTH']][alarm_variables['BP']] = 1
        
        if 'GPT4' in self.LLMs:
            # GPT4 knowledge
            self.prior_knowledge['Alarm']['GPT4'] = np.ones((37, 37)) * 2
            # Adding the specified edges to the GPT4 knowledge dictionary
            self.prior_knowledge['Alarm']['GPT4'][alarm_variables['LVFAILURE']][alarm_variables['PCWP']] = 1
            self.prior_knowledge['Alarm']['GPT4'][alarm_variables['LVFAILURE']][alarm_variables['CO']] = 1
            self.prior_knowledge['Alarm']['GPT4'][alarm_variables['HYPOVOLEMIA']][alarm_variables['CVP']] = 1
            self.prior_knowledge['Alarm']['GPT4'][alarm_variables['HYPOVOLEMIA']][alarm_variables['PCWP']] = 1
            self.prior_knowledge['Alarm']['GPT4'][alarm_variables['HYPOVOLEMIA']][alarm_variables['STROKEVOLUME']] = 1
            self.prior_knowledge['Alarm']['GPT4'][alarm_variables['STROKEVOLUME']][alarm_variables['CO']] = 1
            self.prior_knowledge['Alarm']['GPT4'][alarm_variables['CO']][alarm_variables['BP']] = 1
            self.prior_knowledge['Alarm']['GPT4'][alarm_variables['HR']][alarm_variables['CO']] = 1
            self.prior_knowledge['Alarm']['GPT4'][alarm_variables['TPR']][alarm_variables['BP']] = 1
            self.prior_knowledge['Alarm']['GPT4'][alarm_variables['ARTCO2']][alarm_variables['EXPCO2']] = 1
            self.prior_knowledge['Alarm']['GPT4'][alarm_variables['VENTLUNG']][alarm_variables['EXPCO2']] = 1
            self.prior_knowledge['Alarm']['GPT4'][alarm_variables['INTUBATION']][alarm_variables['VENTLUNG']] = 1
            self.prior_knowledge['Alarm']['GPT4'][alarm_variables['MINVOL']][alarm_variables['VENTLUNG']] = 1
            self.prior_knowledge['Alarm']['GPT4'][alarm_variables['FIO2']][alarm_variables['SAO2']] = 1
            self.prior_knowledge['Alarm']['GPT4'][alarm_variables['SHUNT']][alarm_variables['SAO2']] = 1
            self.prior_knowledge['Alarm']['GPT4'][alarm_variables['PULMEMBOLUS']][alarm_variables['PAP']] = 1

        if 'Gemini' in self.LLMs:
            self.prior_knowledge['Alarm']['Gemini'] = np.ones((37, 37)) * 2
            # Adding the specified edges to the GPT4 knowledge dictionary based on the provided structure and new edges
            self.prior_knowledge['Alarm']['Gemini'][alarm_variables['LVFAILURE']][alarm_variables['ERRLOWOUTPUT']] = 1
            self.prior_knowledge['Alarm']['Gemini'][alarm_variables['HYPOVOLEMIA']][alarm_variables['CVP']] = 1  
            self.prior_knowledge['Alarm']['Gemini'][alarm_variables['CVP']][alarm_variables['ERRLOWOUTPUT']] = 1
            self.prior_knowledge['Alarm']['Gemini'][alarm_variables['LVFAILURE']][alarm_variables['PCWP']] = 1
            self.prior_knowledge['Alarm']['Gemini'][alarm_variables['PULMEMBOLUS']][alarm_variables['PAP']] = 1
            self.prior_knowledge['Alarm']['Gemini'][alarm_variables['ERRLOWOUTPUT']][alarm_variables['HR']] = 1
            self.prior_knowledge['Alarm']['Gemini'][alarm_variables['HR']][alarm_variables['HRBP']] = 1
            self.prior_knowledge['Alarm']['Gemini'][alarm_variables['INSUFFANESTH']][alarm_variables['HR']] = 1  
            self.prior_knowledge['Alarm']['Gemini'][alarm_variables['SHUNT']][alarm_variables['SAO2']] = 1  
            self.prior_knowledge['Alarm']['Gemini'][alarm_variables['FIO2']][alarm_variables['SAO2']] = 1
            self.prior_knowledge['Alarm']['Gemini'][alarm_variables['PULMEMBOLUS']][alarm_variables['VENTLUNG']] = 1 
            self.prior_knowledge['Alarm']['Gemini'][alarm_variables['INTUBATION']][alarm_variables['VENTLUNG']] = 1 
            self.prior_knowledge['Alarm']['Gemini'][alarm_variables['DISCONNECT']][alarm_variables['ERRLOWOUTPUT']] = 1
            self.prior_knowledge['Alarm']['Gemini'][alarm_variables['CATECHOL']][alarm_variables['HR']] = 1
            self.prior_knowledge['Alarm']['Gemini'][alarm_variables['KINKEDTUBE']][alarm_variables['VENTLUNG']] = 1  



    def add_Child_knowledge(self):
        items = ["DuctFlow", "HypDistrib", "CardiacMixing", "HypoxiaInO2", "LungParench", "CO2", "ChestXray", "LungFlow", "Grunting", "Sick", "LVH", "LVHreport", "LowerBodyO2", "RUQO2", "CO2Report", "XrayReport", "BirthAsphyxia", "Disease", "GruntingReport", "Age"]
        child_variables = {item: index for index, item in enumerate(items)}

        self.prior_knowledge['Child'] = {}
        if 'GPT3' in self.LLMs:
            # GPT3 knowledge
            self.prior_knowledge['Child']['GPT3'] = np.ones((20, 20)) * 2
            self.prior_knowledge['Child']['GPT3'][child_variables['CardiacMixing']][child_variables['HypoxiaInO2']] = 1
            self.prior_knowledge['Child']['GPT3'][child_variables['HypDistrib']][child_variables['HypoxiaInO2']] = 1
            self.prior_knowledge['Child']['GPT3'][child_variables['LungParench']][child_variables['HypoxiaInO2']] = 1
            self.prior_knowledge['Child']['GPT3'][child_variables['CO2']][child_variables['Grunting']] = 1
            self.prior_knowledge['Child']['GPT3'][child_variables['Grunting']][child_variables['Sick']] = 1
            self.prior_knowledge['Child']['GPT3'][child_variables['LVH']][child_variables['LVHreport']] = 1
            self.prior_knowledge['Child']['GPT3'][child_variables['LowerBodyO2']][child_variables['HypoxiaInO2']] = 1
            self.prior_knowledge['Child']['GPT3'][child_variables['RUQO2']][child_variables['HypoxiaInO2']] = 1
            self.prior_knowledge['Child']['GPT3'][child_variables['BirthAsphyxia']][child_variables['Disease']] = 1
            self.prior_knowledge['Child']['GPT3'][child_variables['Age']][child_variables['Disease']] = 1


        if 'GPT4' in self.LLMs:
            # GPT4 knowledge
            self.prior_knowledge['Child']['GPT4'] = np.ones((20, 20)) * 2
            self.prior_knowledge['Child']['GPT4'][child_variables['BirthAsphyxia']][child_variables['HypoxiaInO2']] = 1
            self.prior_knowledge['Child']['GPT4'][child_variables['BirthAsphyxia']][child_variables['LungParench']] = 1
            self.prior_knowledge['Child']['GPT4'][child_variables['HypoxiaInO2']][child_variables['Grunting']] = 1
            self.prior_knowledge['Child']['GPT4'][child_variables['LungParench']][child_variables['LungFlow']] = 1
            self.prior_knowledge['Child']['GPT4'][child_variables['LungFlow']][child_variables['LowerBodyO2']] = 1
            self.prior_knowledge['Child']['GPT4'][child_variables['CO2']][child_variables['CO2Report']] = 1
            self.prior_knowledge['Child']['GPT4'][child_variables['ChestXray']][child_variables['XrayReport']] = 1
            self.prior_knowledge['Child']['GPT4'][child_variables['LVH']][child_variables['LVHreport']] = 1
            self.prior_knowledge['Child']['GPT4'][child_variables['Grunting']][child_variables['GruntingReport']] = 1
            self.prior_knowledge['Child']['GPT4'][child_variables['Disease']][child_variables['Sick']] = 1
            

        if 'Gemini' in self.LLMs:
            self.prior_knowledge['Child']['Gemini'] = np.ones((20, 20)) * 2
            self.prior_knowledge['Child']['Gemini'][child_variables['BirthAsphyxia']][child_variables['HypoxiaInO2']] = 1
            self.prior_knowledge['Child']['Gemini'][child_variables['CardiacMixing']][child_variables['HypoxiaInO2']] = 1
            self.prior_knowledge['Child']['Gemini'][child_variables['LungParench']][child_variables['LungFlow']] = 1
            self.prior_knowledge['Child']['Gemini'][child_variables['LungFlow']][child_variables['HypoxiaInO2']] = 1
            self.prior_knowledge['Child']['Gemini'][child_variables['HypoxiaInO2']][child_variables['CO2']] = 1
            self.prior_knowledge['Child']['Gemini'][child_variables['Age']][child_variables['DuctFlow']] = 1

    def add_Earthquake_knowledge(self):
        self.prior_knowledge['Earthquake'] = {}
        if 'GPT3' in self.LLMs:
            # GPT3 knowledge
            self.prior_knowledge['Earthquake']['GPT3'] = np.ones((5, 5)) * 2
            # Burglary → Alarm
            self.prior_knowledge['Earthquake']['GPT3'][0][1] = 1
            # Earthquake → Alarm
            self.prior_knowledge['Earthquake']['GPT3'][2][1] = 1
            # Alarm → JohnCalls
            self.prior_knowledge['Earthquake']['GPT3'][1][3] = 1
            # Alarm → MaryCalls
            self.prior_knowledge['Earthquake']['GPT3'][1][4] = 1

        if 'GPT4' in self.LLMs:
            # Initialize the matrix for 'EarthquakeModel'
            self.prior_knowledge['Earthquake']['GPT4'] = np.ones((5, 5)) * 2
            # Burglary → Alarm
            self.prior_knowledge['Earthquake']['GPT4'][0][1] = 1
            # Earthquake → Alarm
            self.prior_knowledge['Earthquake']['GPT4'][2][1] = 1
            # Alarm → JohnCalls
            self.prior_knowledge['Earthquake']['GPT4'][1][3] = 1
            # Alarm → MaryCalls
            self.prior_knowledge['Earthquake']['GPT4'][1][4] = 1

        if 'Gemini' in self.LLMs:
            self.prior_knowledge['Earthquake'] ['Gemini'] = np.ones((5, 5)) * 2
            # Earthquake → Alarm (if earthquake alarm system exists)
            self.prior_knowledge['Earthquake']['Gemini'][2][1] = 1  # Earthquake -> Alarm
        
    def add_Survey_true_knowledge(self):
        self.prior_knowledge['Survey']['True'] = np.zeros((6, 6))
        self.prior_knowledge['Survey']['True'][0][1] = 1
        self.prior_knowledge['Survey']['True'][2][1] = 1
        self.prior_knowledge['Survey']['True'][1][3] = 1
        self.prior_knowledge['Survey']['True'][1][4] = 1
        self.prior_knowledge['Survey']['True'][3][5] = 1
        self.prior_knowledge['Survey']['True'][4][5] = 1

    # A E S O R T
    def add_Survey_knowledge(self):
        self.prior_knowledge['Survey'] = {}
        if 'GPT3' in self.LLMs:
            # GPT3 knowledge
            self.prior_knowledge['Survey']['GPT3'] = np.ones((6, 6)) * 2
            self.prior_knowledge['Survey']['GPT3'][0][3] = 1
            self.prior_knowledge['Survey']['GPT3'][1][3] = 1
            self.prior_knowledge['Survey']['GPT3'][2][3] = 1
            self.prior_knowledge['Survey']['GPT3'][4][5] = 1
            self.prior_knowledge['Survey']['GPT3'][0][5] = 1
        
        if 'GPT4' in self.LLMs:
            # GPT4 knowledge
            self.prior_knowledge['Survey']['GPT4'] = np.ones((6, 6)) * 2
            self.prior_knowledge['Survey']['GPT4'][2][3] = 1
            self.prior_knowledge['Survey']['GPT4'][3][4] = 1

        if 'Gemini' in self.LLMs:
            # Gemini knowledge
            self.prior_knowledge['Survey']['Gemini'] = np.ones((6, 6)) * 2
            self.prior_knowledge['Survey']['Gemini'][1][3] = 1
            self.prior_knowledge['Survey']['Gemini'][4][3] = 1




    def add_LUCAS_true_knowledge(self):
        self.prior_knowledge['LUCAS'] = {}

        adjmatrix = np.zeros((12, 12))
        adjmatrix[0][1] = 1
        adjmatrix[0][11] = 1
        adjmatrix[2][0] = 1
        adjmatrix[3][0] = 1
        adjmatrix[4][11] = 1
        adjmatrix[4][5] = 1
        adjmatrix[5][7] = 1
        adjmatrix[8][7] = 1
        adjmatrix[9][10] = 1
        adjmatrix[10][8] = 1
        adjmatrix[11][10] = 1
        adjmatrix[11][8] = 1


        self.prior_knowledge['SACHS']['GPT3'] = adjmatrix
        self.prior_knowledge['SACHS']['GPT4'] = adjmatrix
        self.prior_knowledge['SACHS']['Gemini'] = adjmatrix

    def add_Asia_true_knowledge(self):
        self.prior_knowledge['Asia'] = {}


        adjmatrix = np.zeros((8, 8))
        adjmatrix[0][1] = 1
        adjmatrix[1][4] = 1
        adjmatrix[2][3] = 1
        adjmatrix[2][6] = 1
        adjmatrix[3][4] = 1
        adjmatrix[4][5] = 1
        adjmatrix[4][7] = 1
        adjmatrix[6][7] = 1

        self.prior_knowledge['SACHS']['GPT3'] = adjmatrix
        self.prior_knowledge['SACHS']['GPT4'] = adjmatrix
        self.prior_knowledge['SACHS']['Gemini'] = adjmatrix

    def add_SACHS_true_knowledge(self):
        self.prior_knowledge['SACHS'] = {}

        adjmatrix = np.zeros((11, 11))
        adjmatrix[8][1] = 1
        adjmatrix[8][0] = 1
        adjmatrix[8][7] = 1
        adjmatrix[8][10] = 1
        adjmatrix[8][9] = 1
        adjmatrix[7][0] = 1
        adjmatrix[7][1] = 1
        adjmatrix[7][5] = 1
        adjmatrix[7][6] = 1
        adjmatrix[7][10] = 1
        adjmatrix[7][9] = 1

        adjmatrix[0][1] = 1
        adjmatrix[1][5] = 1
        adjmatrix[5][6] = 1
        adjmatrix[2][3] = 1
        adjmatrix[2][4] = 1
        adjmatrix[4][3] = 1

        self.prior_knowledge['SACHS']['GPT3'] = adjmatrix
        self.prior_knowledge['SACHS']['GPT4'] = adjmatrix
        self.prior_knowledge['SACHS']['Gemini'] = adjmatrix

    def add_LUCAS_knowledge(self):
        self.prior_knowledge['LUCAS'] = {}

        if 'GPT3' in self.LLMs:
            # GPT3 knowledge
            self.prior_knowledge['LUCAS']['GPT3'] = np.ones((12, 12)) / 2
            self.prior_knowledge['LUCAS']['GPT3'][2][0] = 1
            self.prior_knowledge['LUCAS']['GPT3'][3][0] = 1
            self.prior_knowledge['LUCAS']['GPT3'][0][1] = 1
            self.prior_knowledge['LUCAS']['GPT3'][0][11] = 1
            self.prior_knowledge['LUCAS']['GPT3'][4][11] = 1
            self.prior_knowledge['LUCAS']['GPT3'][9][0] = 1
            self.prior_knowledge['LUCAS']['GPT3'][10][11] = 1
            self.prior_knowledge['LUCAS']['GPT3'][8][0] = 1
            self.prior_knowledge['LUCAS']['GPT3'][5][0] = 1

        if 'GPT4' in self.LLMs:
            # GPT4 knowledge
            self.prior_knowledge['LUCAS']['GPT4'] = np.ones((12, 12)) / 2
            self.prior_knowledge['LUCAS']['GPT4'][3][0] = 1
            self.prior_knowledge['LUCAS']['GPT4'][0][1] = 1
            self.prior_knowledge['LUCAS']['GPT4'][0][11] = 1
            self.prior_knowledge['LUCAS']['GPT4'][4][11] = 1
            self.prior_knowledge['LUCAS']['GPT4'][0][10] = 1
            self.prior_knowledge['LUCAS']['GPT4'][5][0] = 1

        if 'Gemini' in self.LLMs:
            # Gemini knowledge
            self.prior_knowledge['LUCAS']['Gemini'] = np.ones((12, 12)) / 2
            self.prior_knowledge['LUCAS']['Gemini'][3][0] = 1
            self.prior_knowledge['LUCAS']['Gemini'][0][1] = 1
            self.prior_knowledge['LUCAS']['Gemini'][4][11] = 1
            self.prior_knowledge['LUCAS']['Gemini'][11][10] = 1
            self.prior_knowledge['LUCAS']['Gemini'][11][8] = 1

    def add_Asia_knowledge(self):
        self.prior_knowledge['Asia'] = {}

        if 'GPT3' in self.LLMs:
            # GPT3 knowledge
            self.prior_knowledge['Asia']['GPT3'] = np.ones((8, 8)) / 2
            self.prior_knowledge['Asia']['GPT3'][2][3] = 1
            self.prior_knowledge['Asia']['GPT3'][3][4] = 1
            self.prior_knowledge['Asia']['GPT3'][4][5] = 1
            self.prior_knowledge['Asia']['GPT3'][4][6] = 1
            self.prior_knowledge['Asia']['GPT3'][4][7] = 1
            self.prior_knowledge['Asia']['GPT3'][1][4] = 1
            self.prior_knowledge['Asia']['GPT3'][6][7] = 1
            self.prior_knowledge['Asia']['GPT3'][7][5] = 1

        if 'GPT4' in self.LLMs:
            # GPT4 knowledge
            self.prior_knowledge['Asia']['GPT4'] = np.ones((8, 8)) / 2
            self.prior_knowledge['Asia']['GPT4'][0][1] = 1
            self.prior_knowledge['Asia']['GPT4'][1][4] = 1
            self.prior_knowledge['Asia']['GPT4'][2][3] = 1
            self.prior_knowledge['Asia']['GPT4'][2][6] = 1
            self.prior_knowledge['Asia']['GPT4'][3][4] = 1
            self.prior_knowledge['Asia']['GPT4'][4][5] = 1
            self.prior_knowledge['Asia']['GPT4'][4][7] = 1
            self.prior_knowledge['Asia']['GPT4'][6][7] = 1


        if 'Gemini' in self.LLMs:
            # Gemini knowledge
            self.prior_knowledge['Asia']['Gemini'] = np.ones((8, 8)) / 2
            self.prior_knowledge['Asia']['Gemini'][1][4] = 1
            self.prior_knowledge['Asia']['Gemini'][2][3] = 1
            self.prior_knowledge['Asia']['Gemini'][3][4] = 1
            self.prior_knowledge['Asia']['Gemini'][3][7] = 1
            self.prior_knowledge['Asia']['Gemini'][6][7] = 1



    def add_SACHS_knowledge(self):
        self.prior_knowledge['SACHS'] = {}

        if 'GPT3' in self.LLMs:
            # GPT3 knowledge
            self.prior_knowledge['SACHS']['GPT3'] = np.ones((11, 11)) * 0
            self.prior_knowledge['SACHS']['GPT3'][2][8] = 1
            self.prior_knowledge['SACHS']['GPT3'][2][3] = 1
            self.prior_knowledge['SACHS']['GPT3'][3][4] = 1
            self.prior_knowledge['SACHS']['GPT3'][4][6] = 1
            self.prior_knowledge['SACHS']['GPT3'][4][7] = 1
            self.prior_knowledge['SACHS']['GPT3'][5][6] = 1
            self.prior_knowledge['SACHS']['GPT3'][5][10] = 1
            self.prior_knowledge['SACHS']['GPT3'][6][9] = 1
            self.prior_knowledge['SACHS']['GPT3'][7][5] = 1
            self.prior_knowledge['SACHS']['GPT3'][7][6] = 1
            self.prior_knowledge['SACHS']['GPT3'][7][9] = 1
            self.prior_knowledge['SACHS']['GPT3'][8][2] = 1
            self.prior_knowledge['SACHS']['GPT3'][8][5] = 1
            self.prior_knowledge['SACHS']['GPT3'][8][6] = 1
            self.prior_knowledge['SACHS']['GPT3'][8][7] = 1
            self.prior_knowledge['SACHS']['GPT3'][9][10] = 1

        if 'GPT4' in self.LLMs:
            # GPT4 knowledge
            self.prior_knowledge['SACHS']['GPT4'] = np.ones((11, 11)) * 0
            self.prior_knowledge['SACHS']['GPT4'][2][8] = 1
            self.prior_knowledge['SACHS']['GPT4'][2][3] = 1
            self.prior_knowledge['SACHS']['GPT4'][3][4] = 1
            self.prior_knowledge['SACHS']['GPT4'][4][6] = 1
            self.prior_knowledge['SACHS']['GPT4'][0][1] = 1
            self.prior_knowledge['SACHS']['GPT4'][1][5] = 1


        if 'Gemini' in self.LLMs:
            # Gemini knowledge
            self.prior_knowledge['SACHS']['Gemini'] = np.ones((11, 11)) * 0
            self.prior_knowledge['SACHS']['Gemini'][2][3] = 1
            self.prior_knowledge['SACHS']['Gemini'][3][4] = 1
            self.prior_knowledge['SACHS']['Gemini'][4][6] = 1
            self.prior_knowledge['SACHS']['Gemini'][0][1] = 1
            self.prior_knowledge['SACHS']['Gemini'][1][5] = 1
            self.prior_knowledge['SACHS']['Gemini'][8][10] = 1