import pandas
import numpy
from pyDecision.algorithm import bw_method
from sklearn.preprocessing import MinMaxScaler
import mcdm



def effective_risk_matrix_calculation(cei_risk,object_risk,cei_failures):
    '''
    This function achieves two tasks.Firstly, it calculates the effective risk each cei has on the object type. This is achieved by doing a cross product of cei severity on object severity. Secondly, it calculates the effective risk matrix by multiplying the risk matrix with the corresponding cei_failures.

    It takes 3 arguments

    cei_risk -> A dictionary containing the severity associated with each CEI in numeric form.

    object_risk -> A dictionary containing the severity associated with each object_type in numeric form.

    cei_failures -> An aggregate matrix derived from the CEI metrics. It indicates what percentage of each object type has failed each CEI.

    '''
    object_risk = pandas.DataFrame(object_risk,index=[0])
    cei_risk = pandas.DataFrame(cei_risk,index=[0])
    
    risk_matrix = pandas.DataFrame(cei_risk.T.values *object_risk.values )
    risk_matrix.index = cei_risk.columns
    risk_matrix.columns = object_risk.columns
  
    return (risk_matrix.T*cei_failures).fillna(0.0)

def derive_cei_preference(cei_severity):

    '''
    Return MIC and LIC values based on CEI Severity. Essentially it ranks the severity values from least to most important and vice-versa. These parameters are used in the BWM calculation.

    The function takes only cei_severity as input.
    '''
    cei_severity = pandas.DataFrame(cei_severity,index=["CEI Severity"]).T
    mic = cei_severity["CEI Severity"].rank(method="dense",ascending=False).values
    lic = cei_severity["CEI Severity"].rank(method="dense",ascending=True).values
    return mic,lic

def derive_cei_weights_bw(risk_matrix,cei_severity,size=15,iterations=150):
    ''' 
    This function computes the weights for each CEI. These weights maybe provided as an input to the priority_rank function described below.
    More about the BW Method can be found here: https://prevalentai.atlassian.net/wiki/spaces/DS/pages/2885451828/Best+Worst+Method+BWM
    You may find the original paper here :https://prevalentindia-my.sharepoint.com/:b:/g/personal/hariraj_k_prevalent_ai/Ed1UgTmU59xFgpX-Z8GyNwUBwfmtbc_S1tVnI4rGbSnUZg?e=PkN7KC

    It takes the following arguments:
    risk_matrix -> The matrix containing the associated risk for each CEI based on the desired attribute.
    cei_severity -> The risk score associated with each CEI.
    '''
    risk_values = risk_matrix.values
    mic,lic = derive_cei_preference(cei_severity)
    weights =  bw_method(risk_values, mic, lic, size = size, iterations = iterations)
    return weights

def object_priority_rank(risk_matrix,weights):
    '''
    This function takes the risk matrix and criteria (cei weights) as input. The associated priority is determined using.
    More information on TOPSIS can be found here : https://prevalentai.atlassian.net/wiki/spaces/DS/pages/2881029139/TOPSIS
    
    '''
    risk_values = risk_matrix.values
    min_max_scaler = MinMaxScaler()
    risk_scaled = min_max_scaler.fit_transform(risk_values)

    priority_rank = mcdm.rank(risk_scaled, w_vector=weights, s_method="TOPSIS",alt_names=risk_matrix.index)

    return priority_rank


def priority_score_calculation(weights,priority_rank,risk_matrix,device_column_name,cei_column_name):

    '''
    The function takes the weights and priority_rank as input, it multiplies the priority score for each alternative with the weights for criteria to calculate the a score for each alternative-criteria combination.
    It takes the following inputs:
        
        weights ->  weights for each criteria
        priority_rank -> priorirty score associated with each alternative
        risk_matrix -> The risk matrix
        device_column_name -> The column name containing the alternative value in the evidence table
        cei_column_name -> The column name containing the CEI code in the evidence table
    '''
    priority_score_column = "priority_score"
    temp_column_name = "asset_cei"
    criteria_names = risk_matrix.columns
    alternative_names = risk_matrix.index
    
    priority_rank_column = "priority_rank"
    failure_column = "failure_count"
    priority_scores =[]
    
    # Multiplying weights and priority to derive alternative-crtieria ranking. Log Transformation is used here to account for any skewness that may arise.
    
    for i in range(0, len(priority_rank)):
        for j in range(0, len(weights)):
            priority_scores.append(-1/numpy.log10((priority_rank[i][1])*weights[j]))
    
    priority_key = []
    for i in range(0, len(alternative_names)):
        for j in range(0, len(criteria_names)):
             priority_key.append(alternative_names[i] + ":" + criteria_names[j])

    list_zip = zip(priority_key, priority_scores)
    zipped_list = list(list_zip)
    
    sorted_priority = sorted(zipped_list, key=lambda x: x[1], reverse=True )

    priority_df =  pandas.DataFrame(sorted_priority,columns=[temp_column_name,priority_score_column])
    priority_df[[device_column_name,cei_column_name]] = priority_df[temp_column_name].apply(lambda x: pandas.Series(x.split(":")))
    
    risk_matrix_unstacked = risk_matrix.unstack().reset_index()
    
    risk_matrix_unstacked.columns = [cei_column_name,device_column_name,failure_column]
    priority_df = priority_df.merge(risk_matrix_unstacked,on=[cei_column_name,device_column_name])
    priority_df = priority_df[priority_df[failure_column] > 0]
    priority_df[priority_rank_column] = priority_df[priority_score_column].rank(ascending=False,method="dense")
    return priority_df[[device_column_name,cei_column_name,priority_rank_column, priority_score_column]]


def priority_rank(host_severity,cei_severity,cei_failures,object_type_column,cei_code_column):
    '''
    Deriving the priority rank from the host severity ,cei severity and the cei failures. It returns a dataframe that gaves the  priority rank for object-cei combination.
    
    host_severity -> A dictionary containing the severity status of a host
    cei_severity -> A dictionary containing the severity status of a cei
    
    cei_failures -> A dataframe containing the failure percentage of each cei-object type combination
    object_type_column -> Column name containing the required object type
    
    cei_code_column -> Column name containing the required cei code
    
    
   
'''
    
    risk_matrix = effective_risk_matrix_calculation(cei_severity,host_severity,cei_failures)
    cei_severity = {cei:cei_severity[cei] for cei in risk_matrix.columns}
    
    mic,lic = derive_cei_preference(cei_severity)
    cei_weights = derive_cei_weights_bw(risk_matrix,cei_severity)
    scores = object_priority_rank(risk_matrix,cei_weights)
    priority_df = priority_score_calculation(cei_weights,scores,risk_matrix,object_type_column,cei_code_column)
    
    return priority_df

def insights (cei_metric_counts,cei_recc_dict,cei_status_column, attribute_columns, attribute_names, cei_code_column, object_type_column, entity_column, priority_rank_column ,impact_column ):
    '''
    Generation of insights based on the priority score and rank and based on the required attribute . 
    It takes the following arguments. 
    
    
    cei_metric_counts -> CEI Metric containing the priority rank of the cei code and object type.
    cei_recc_dict -> Contains the format for the reccomendation string.
    cei_status_column -> The field containing the Status of the CEI .
    attribute_columns -> List of column names containing the attributes required for the insights.
    attribute_names -> List containing attribute names.
    cei_code_column -> The name of the column containing the cei code.
    object_type_column -> The name of the column containing the object type.
    entity_column -> The name of the column containing the entity type.
    priority_rank_column -> The name of the column containing the priority rank.
    impact_column -> The column name required for the impact score.
    
    '''
    reccomendation_column = "reccomendation"
    inscope = cei_metric_counts.groupby([cei_code_column,object_type_column])[entity_column].value_counts()
    inscope.name = "total_scope"
    inscope = inscope.reset_index()
    
    cei_metric_counts = cei_metric_counts[cei_metric_counts[cei_status_column] == "Failed"]
    insights_df = pandas.DataFrame([],columns=[entity_column,cei_code_column,object_type_column,priority_rank_column,impact_column])
    for idx in range(len(attribute_columns)):
            attribute = attribute_columns[idx]
            temp_df = cei_metric_counts.groupby([cei_code_column,object_type_column,entity_column,priority_rank_column])[attribute].value_counts()
            temp_df.name = "failure_count"
            temp_df = temp_df.reset_index()
            temp_df = temp_df.rename(columns={attribute:"attribute_value"})
            temp_df["attribute"] = attribute_names[idx]
            insights_df = pandas.concat([insights_df,temp_df])
    insights_df = insights_df.merge(inscope,on=[cei_code_column,object_type_column,entity_column])
    
    insights_df[impact_column] = 100*insights_df["failure_count"]/insights_df["total_scope"]
    
    insights_df = insights_df.sort_values(impact_column,ascending=False).sort_values(priority_rank_column)
    
    insights_df[reccomendation_column] = insights_df[cei_code_column].apply(lambda x: cei_recc_dict[x])
    insights_df[reccomendation_column] =  insights_df.apply(lambda row: row[reccomendation_column].replace("($object_type)",row[object_type_column]).replace("($attribute)",row["attribute_value"]),axis=1)
    
    return insights_df
