all_tasks = {}

task_subgroup_1 = {}




template = {}

template['source']="{} is connected with {} within one hop. Will {} be connected with {} within one hop?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-1-1-1"

task_subgroup_1["1-1-1-1"] = template




template = {}

template['source']="{} is connected with {} within one hop. Which other node will be connected to {} within one hop?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 3
template['source_argv'] = ['node_id', 'node_id_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-1-1-2"

task_subgroup_1["1-1-1-2"] = template

template = {}

template['source']="{} linked to nodes within one hop in the following order: {}. Will {} be the next node to be connected to {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-1-1-3"

task_subgroup_1["1-1-1-3"] = template

template = {}

template['source']="{} linked to nodes within one hop in the following order: {}. Which node is most likely to be the next node connected to {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 3
template['source_argv'] = ['node_id', 'node_id_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-1-1-4"

task_subgroup_1["1-1-1-4"] = template

template = {}

template['source']="{} is connected with {} within two hops. Will {} be connected to {} within two hops?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-1-2-1"

task_subgroup_1["1-1-2-1"] = template

template = {}

template['source']="{} is connected with {} within two hops. Which other node will be connected to {} within two hops?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 3
template['source_argv'] = ['node_id', 'node_id_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-1-2-2"

task_subgroup_1["1-1-2-2"] = template

template = {}

template['source']="{} is connected with {} within two hops through {}, respectively. Will {} be connected to {} within two hops through {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 6
template['source_argv'] = ['node_id','node_id_list','node_id_list','node_id','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-1-2-3"

task_subgroup_1["1-1-2-3"] = template

template = {}

template['source']="{} is connected with {} within two hops through {}, respectively. Which other node will be connected to {} within two hops through {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','node_id_list','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-1-2-4"

task_subgroup_1["1-1-2-4"] = template

template = {}


template['source']="{} is connected with {} within three hops. Will {} be connected with {} within three hops?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-1-3-1"

task_subgroup_1["1-1-3-1"] = template

template = {}

template['source']="{} is connected with {} within three hops. Which other node will be connected to {} within three hops?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 3
template['source_argv'] = ['node_id', 'node_id_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-1-3-2"

task_subgroup_1["1-1-3-2"] = template

template = {}

template['source']="{} is connected with {} within three hops through {}, respectively. Will {} be connected to {} within three hops through {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 6
template['source_argv'] = ['node_id','node_id_list','node_id_tuple_list','node_id','node_id','node_id_tuple']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-1-3-3"

task_subgroup_1["1-1-3-3"] = template

template = {}

template['source']="{} is connected with {} within three hops through {}, respectively. Which other node will be connected to {} within three hops through {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','node_id_tuple_list','node_id','node_id_tuple']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-1-3-4"

task_subgroup_1["1-1-3-4"] = template



###
template = {}

template['source']="{} is connected with {} within one hop through featured edges: {}, respectively. Will {} be connected to {} within one hop through featured edge: {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 6
template['source_argv'] = ['node_id', 'node_id_list','edge_feature_list','node_id','node_id','edge_feature']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-2-1-1"

task_subgroup_1["1-2-1-1"] = template



template = {}

template['source']="{} is connected with {} within one hop through featured edges: {}, respectively. Which other node will be connected to {} within one hop through featured edge: {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','edge_feature_list','node_id','edge_feature']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-2-1-2"

task_subgroup_1["1-2-1-2"] = template

template = {}

template['source']="{} linked to nodes within one hop in the following order: {} through the following featured edges: {}. Will {} be the next node to be connected to {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','edge_feature_list','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-2-1-3"

task_subgroup_1["1-2-1-3"] = template

template = {}

template['source']="{} linked to nodes within one hop in the following order: {} throught the following featured edges: {}. Which node is most likely to be the next node connected to {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','edge_feature_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-2-1-4"

task_subgroup_1["1-2-1-4"] = template

template = {}

template['source']="{} is connected with {} within two hops through featured paths: {}, respectively. Will {} be connected to {} within two hops through featured path: {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 6
template['source_argv'] = ['node_id', 'node_id_list','feature_tuple_list','node_id','node_id','feature_tuple']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-2-2-1"

task_subgroup_1["1-2-2-1"] = template

template = {}

template['source']="{} is connected with {} within two hops through featured paths: {}, respectively. Which other node will be connected to {} within two hops through featured path: {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','feature_tuple_list','node_id','feature_tuple']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-2-2-2"

task_subgroup_1["1-2-2-2"] = template

template = {}

template['source']="{} is connected with {} within two hops through {} and featured paths: {}, respectively. Will {} be connected to {} within two hops through {} and featured path: {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 8
template['source_argv'] = ['node_id','node_id_list','node_id_list','feature_tuple_list','node_id','node_id','node_id','feature_tuple']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-2-2-3"

task_subgroup_1["1-2-2-3"] = template

template = {}

template['source']="{} is connected with {} within two hops through {} and featured paths: {}, respectively. Which other node will be connected to {} within two hops through {} and featured path: {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 7
template['source_argv'] = ['node_id','node_id_list','node_id_list','feature_tuple_list','node_id','node_id','feature_tuple']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-2-2-4"

task_subgroup_1["1-2-2-4"] = template

template = {}


template['source']="{} is connected with {} within three hops through featured paths: {}, respectively. Will {} be connected to {} within three hop through featured path: {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 6
template['source_argv'] = ['node_id', 'node_id_list','feature_triple_list','node_id','node_id','feature_triple']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-2-3-1"

task_subgroup_1["1-2-3-1"] = template

template = {}

template['source']="{} is connected with {} within three hops through featured paths: {}, respectively. Which other node will be connected to {} within three hops through featured path: {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','feature_triple_list','node_id','feature_triple']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-2-3-2"

task_subgroup_1["1-2-3-2"] = template

template = {}

template['source']="{} is connected with {} within three hops through {} and featured paths: {}, respectively. Will {} be connected to {} within three hops through {} and featured path: {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 8
template['source_argv'] = ['node_id','node_id_list','node_id_tuple_list','feature_triple_list','node_id','node_id','node_id_tuple','feature_triple']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-2-3-3"

task_subgroup_1["1-2-3-3"] = template

template = {}

template['source']="{} is connected with {} within three hops through {} and featured paths {}, respectively. Which other node will be connected to {} within three hops through {} and fetured path: {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 7
template['source_argv'] = ['node_id','node_id_list','node_id_tuple_list','feature_triple_list','node_id','node_id_tuple','feature_triple']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-2-3-4"

task_subgroup_1["1-2-3-4"] = template

##



template = {}

template['source']="({},{}) is connected with {} within one hop. Will ({},{}) be connected with ({},{}) within one hop?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 7
template['source_argv'] = ['node_id', 'node_id_list','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-3-1-1"

task_subgroup_1["1-3-1-1"] = template




template = {}

template['source']="({},{}) is connected with {} within one hop. Which other node will be connected to ({},{}) within one hop?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-3-1-2"

task_subgroup_1["1-3-1-2"] = template


template = {}

template['source']="({},{}) is connected with {} within two hops. Will ({},{}) be connected to ({},{}) within two hops?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 7
template['source_argv'] = ['node_id', 'node_id_list','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-3-2-1"

task_subgroup_1["1-3-2-1"] = template

template = {}

template['source']="({},{}) is connected with {} within two hops. Which other node will be connected to ({},{}) within two hops?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-3-2-2"

task_subgroup_1["1-3-2-2"] = template

template = {}

template['source']="({},{}) is connected with {} within two hops through {}, respectively. Will ({},{}) be connected to ({},{}) within two hops through ({},{})?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 10
template['source_argv'] = ['node_id','node_id_list','node_id_list','node_id','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-3-2-3"

task_subgroup_1["1-3-2-3"] = template

template = {}

template['source']="({},{}) is connected with {} within two hops through {}, respectively. Which other node will be connected to ({},{}) within two hops through ({},{})?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 8
template['source_argv'] = ['node_id', 'node_id_list','node_id_list','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-3-2-4"

task_subgroup_1["1-3-2-4"] = template

template = {}


template['source']="({},{}) is connected with {} within three hops. Will ({},{}) be connected with ({},{}) within three hops?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 7
template['source_argv'] = ['node_id', 'node_id_list','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-3-3-1"

task_subgroup_1["1-3-3-1"] = template

template = {}

template['source']="({},{}) is connected with {} within three hops. Which other node will be connected to ({},{}) within three hops?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-3-3-2"

task_subgroup_1["1-3-3-2"] = template

template = {}

template['source']="({},{}) is connected with {} within three hops through {}, respectively. Will ({},{}) be connected to ({},{}) within three hops through ({},{}; {},{})?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 9
template['source_argv'] = ['node_id','node_id_list','node_id_tuple_list','node_id','node_id','node_id_tuple']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-3-3-3"

task_subgroup_1["1-3-3-3"] = template

template = {}

template['source']="({},{}) is connected with {} within three hops through {}, respectively. Which other node will be connected to ({},{}) within three hops through ({},{}; {},{})?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 7
template['source_argv'] = ['node_id', 'node_id_list','node_id_tuple_list','node_id','node_id_tuple']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-3-3-4"

task_subgroup_1["1-3-3-4"] = template


template = {}

template['source']="Perform Link Prediction for the node: Node represents academic paper with a specific topic. node {} is featured with its abstract: {}. Which other node will be connected to {} within one hop?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 7
template['source_argv'] = ['node_id', 'node_id_list','node_id_tuple_list','node_id','node_id_tuple']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "1-7-7-4"

task_subgroup_1["1-7-7-4"] = template

template = {}

template['source']="Perform Link Prediction for the node: Node represents academic paper with a specific topic. node {} is featured with its abstract: {}. Will node {} be connected to {} within one hop?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 9
template['source_argv'] = ['node_id','node_id_list','node_id_tuple_list','node_id','node_id','node_id_tuple']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-7-7-3"

task_subgroup_1["1-7-7-3"] = template






all_tasks['link'] =  task_subgroup_1



##
task_subgroup_2 = {}

template = {}

template['source']="{} is connected with {} within one hop. Should {} be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','node_id','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-1-1-1"

task_subgroup_2["2-1-1-1"] = template

template = {}

template['source']="{} is connected with {} within one hop. Which category should {} be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 3
template['source_argv'] = ['node_id', 'node_id_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-1-1-2"

task_subgroup_2["2-1-1-2"] = template

template = {}

template['source']="{} linked to nodes within one hop in the following order: {}. Should {} be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','node_id','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-1-1-3"

task_subgroup_2["2-1-1-3"] = template

template = {}

template['source']="{} linked to nodes within one hop in the following order: {}. Which category should {} be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 3
template['source_argv'] = ['node_id', 'node_id_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-1-1-4"

task_subgroup_2["2-1-1-4"] = template

template = {}

template['source']="{} is connected with {} within two hops. Should {} be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','node_id','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-1-2-1"

task_subgroup_2["2-1-2-1"] = template

template = {}

template['source']="{} is connected with {} within two hops. Which category should {} be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 3
template['source_argv'] = ['node_id', 'node_id_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-1-2-2"

task_subgroup_2["2-1-2-2"] = template

template = {}

template['source']="{} is connected with {} within two hops through {}, respectively. Should {} be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 5
template['source_argv'] = ['node_id','node_id_list','node_id_list','node_id','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-1-2-3"

task_subgroup_2["2-1-2-3"] = template

template = {}

template['source']="{} is connected with {} within two hops through {}, respectively. Which category should {} be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','node_id_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-1-2-4"

task_subgroup_2["2-1-2-4"] = template

template = {}


template['source']="{} is connected with {} within three hops. Should {} be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','node_id','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-1-3-1"

task_subgroup_2["2-1-3-1"] = template

template = {}

template['source']="{} is connected with {} within three hops. Which category should {} be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 3
template['source_argv'] = ['node_id', 'node_id_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-1-3-2"

task_subgroup_2["2-1-3-2"] = template

template = {}

template['source']="{} is connected with {} within three hops through {}, respectively. Should {} be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 5
template['source_argv'] = ['node_id','node_id_list','node_id_tuple_list','node_id','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-1-3-3"

task_subgroup_2["2-1-3-3"] = template

template = {}
# (0-5-6-1) (0-7-8-2)
template['source']="{} is connected with {} within three hops through {}, respectively. Which category should {} be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','node_id_tuple_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-1-3-4"

task_subgroup_2["2-1-3-4"] = template




#
template = {}

template['source']="{} is connected with {} within one hop through featured edges: {}, respectively. Should {} be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','edge_feature_list','node_id','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-2-1-1"

task_subgroup_2["2-2-1-1"] = template

template = {}

template['source']="{} is connected with {} within one hop through featured edges: {}, respectively. Which category should {} be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','edge_feature_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-2-1-2"

task_subgroup_2["2-2-1-2"] = template

template = {}

template['source']="{} linked to nodes within one hop in the following order: {} through the following featured edges: {}. Should {} be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','edge_feature_list','node_id','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-2-1-3"

task_subgroup_2["2-2-1-3"] = template

template = {}

template['source']="{} linked to nodes within one hop in the following order: {} through the following featured edges: {}. Which category should {} be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','edge_feature_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-2-1-4"

task_subgroup_2["2-2-1-4"] = template

template = {}

template['source']="{} is connected with {} within two hops through featured paths: {}, respectively. Should {} be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','feature_tuple_list','node_id','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-2-2-1"

task_subgroup_2["2-2-2-1"] = template

template = {}

template['source']="{} is connected with {} within two hops through featured paths: {}, respectively. Which category should {} be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','feature_tuple_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-2-2-2"

task_subgroup_2["2-2-2-2"] = template

template = {}

template['source']="{} is connected with {} within two hops through {} and featured paths: {}, respectively. Should {} be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 6
template['source_argv'] = ['node_id','node_id_list','node_id_list','feature_tuple_list','node_id','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-2-2-3"

task_subgroup_2["2-2-2-3"] = template

template = {}

template['source']="{} is connected with {} within two hops through {} and featured paths: {}, respectively. Which category should {} be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 5
template['source_argv'] = ['node_id','node_id_list','node_id_list','feature_tuple_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-2-2-4"

task_subgroup_2["2-2-2-4"] = template

template = {}


template['source']="{} is connected with {} within three hops through featured paths: {}, respectively. Should {} be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','feature_triple_list','node_id','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-2-3-1"

task_subgroup_2["2-2-3-1"] = template

template = {}

template['source']="{} is connected with {} within three hops through featured paths: {}, respectively. Which category should {} be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','feature_triple_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-2-3-2"

task_subgroup_2["2-2-3-2"] = template

template = {}

template['source']="{} is connected with {} within three hops through {} and featured paths: {}, respectively. Should {} be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 6
template['source_argv'] = ['node_id','node_id_list','node_id_tuple_list','feature_triple_list','node_id','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-2-3-3"

task_subgroup_2["2-2-3-3"] = template

template = {}

template['source']="{} is connected with {} within three hops through {} and featured paths: {}, respectively. Which category should {} be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 5
template['source_argv'] = ['node_id','node_id_list','node_id_tuple_list','feature_triple_list','node_id']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-2-3-4"

task_subgroup_2["2-2-3-4"] = template







#
template = {}
##
template['source']="({},{}) is connected with {} within one hop. Should ({},{}) be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 6
template['source_argv'] = ['node_id','abs', 'node_id_list','node_id','abs','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-3-1-1"

task_subgroup_2["2-3-1-1"] = template

template = {}

template['source']="({},{}) is connected with {} within one hop. Which category should ({},{}) be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 5
template['source_argv'] = ['node_id','abs', 'node_id_list','node_id','abs']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-3-1-2"

task_subgroup_2["2-3-1-2"] = template

#
template = {}

template['source']="({},{}) is connected with {} within two hops. Should ({},{}) be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 6
template['source_argv'] = ['node_id','abs', 'node_id_list','node_id','abs','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-3-2-1"

task_subgroup_2["2-3-2-1"] = template

template = {}

template['source']="({},{}) is connected with {} within two hops. Which category should ({},{}) be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 3
template['source_argv'] = ['node_id','abs', 'node_id_list','node_id','abs']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-3-2-2"

task_subgroup_2["2-3-2-2"] = template

template = {}

template['source']="({},{}) is connected with {} within two hops through {}, respectively. Should ({},{}) be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 7
template['source_argv'] = ['node_id','abs','node_id_list','node_id_list','node_id','abs','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-3-2-3"

task_subgroup_2["2-3-2-3"] = template

template = {}

template['source']="({},{}) is connected with {} within two hops through {}, respectively. Which category should ({},{}) be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 6
template['source_argv'] = ['node_id', 'abs','node_id_list','node_id_list','node_id','abs']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-3-2-4"

task_subgroup_2["2-3-2-4"] = template

template = {}


template['source']="({},{}) is connected with {} within three hops. Should ({},{}) be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 6
template['source_argv'] = ['node_id','abs', 'node_id_list','node_id','abs','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-3-3-1"

task_subgroup_2["2-3-3-1"] = template

template = {}

template['source']="({},{}) is connected with {} within three hops. Which category should ({},{}) be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 5
template['source_argv'] = ['node_id','abs', 'node_id_list','node_id','abs']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-3-3-2"

task_subgroup_2["2-3-3-2"] = template

template = {}

template['source']="({},{}) is connected with {} within three hops through {}, respectively. Should ({},{}) be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 7
template['source_argv'] = ['node_id','abs','node_id_list','node_id_tuple_list','node_id','abs','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-3-3-3"

task_subgroup_2["2-3-3-3"] = template

template = {}

template['source']="({},{}) is connected with {} within three hops through {}, respectively. Which category should ({},{}) be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 6
template['source_argv'] = ['node_id','abs', 'node_id_list','node_id_tuple_list','node_id','abs']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "2-3-3-4"

task_subgroup_2["2-3-3-4"] = template

#



template = {}

template['source']="node represents academic paper with a specific topic. node {} is featured with its abstract: {}. Should {} be classified as {}?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'abs','node_id','attribute']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "5-5-5-5"

task_subgroup_2["5-5-5-5"] = template


template = {}

template['source']="Categorize the article by topic: Node represents academic paper with a specific topic. node ({},{}) is featured with its abstract: {}. Which category should ({},{}) be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 3
template['source_argv'] = ['node_id', 'abs','node_id']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "6-6-6-6"

task_subgroup_2["6-6-6-6"] = template

template = {}

template['source']="Classify the article according to its topic into one of the following categories: [experimental, second, first]. Node represents academic paper with a specific topic. node ({},{}) is featured with its abstract: {}. Which category should ({},{}) be classified as?"
template['target'] = "{}"
template['task'] = "classification"
template['source_argc'] = 3
template['source_argv'] = ['node_id', 'abs','node_id']
template['target_argc'] = 1
template['target_argv'] = ['attribute']
template['id'] = "6-6-6-7"

task_subgroup_2["6-6-6-7"] = template

#





all_tasks['classification'] =  task_subgroup_2
































#####

task_subgroup_3 = {}

template = {}

template['source']="{} is connected with {} within one hop. Now I want {} to be connected to {} in two hops, should {} be the intermediate node?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','node_id','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "3-1-1-1"

task_subgroup_3["3-1-1-1"] = template

template = {}

template['source']="{} is connected with {} within one hop. Now I want {} to be connected to {} in two hops, which intermediate node should I choose?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "3-1-1-2"

task_subgroup_3["3-1-1-2"] = template

template = {}

template['source']="{} linked to nodes within one hop in the following order: {}. Now I want {} to be connected to {} in two hops, should {} be the intermediate node?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','node_id','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "3-1-1-3"

task_subgroup_3["3-1-1-3"] = template

template = {}

template['source']="{} linked to nodes within one hop in the following order: {}. Now I want {} to be connected to {} in two hops, which intermediate node should I choose?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "3-1-1-4"

task_subgroup_3["3-1-1-4"] = template

template = {}

template['source']="{} is connected with {} within two hops. Now I want {} to be connected to {} in two hops, should {} be the intermediate node?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','node_id','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "3-1-2-1"

task_subgroup_3["3-1-2-1"] = template

template = {}

template['source']="{} is connected with {} within two hops. Now I want {} to be connected to {} in two hops, which intermediate node should I choose?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "3-1-2-2"

task_subgroup_3["3-1-2-2"] = template

template = {}

template['source']="{} is connected with {} within two hops through {}, respectively. Now I want {} to be connected to {} in two hops, should {} be the intermediate node?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 6
template['source_argv'] = ['node_id','node_id_list','node_id_list','node_id','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "3-1-2-3"

task_subgroup_3["3-1-2-3"] = template

template = {}

template['source']="{} is connected with {} within two hops through {}, respectively. Now I want {} to be connected to {} in two hops, which intermediate node should I choose?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','node_id_list','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "3-1-2-4"

task_subgroup_3["3-1-2-4"] = template

#给3hop， 问3hop

template = {}

template['source']="{} is connected with {} within three hops. Now I want {} to be connected to {} in three hops, should {} be the intermediate node pair?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','node_id','node_id','node_id_pair']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "3-1-3-1"

task_subgroup_3["3-1-3-1"] = template

template = {}

template['source']="{} is connected with {} within three hops. Now I want {} to be connected to {} in three hops, which intermediate node pair should I choose?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 4
template['source_argv'] = ['node_id', 'node_id_list','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['node_id_pair']
template['id'] = "3-1-3-2"

task_subgroup_3["3-1-3-2"] = template

template = {}

template['source']="{} is connected with {} within three hops through {}, respectively. Now I want {} to be connected to {} in three hops, should {} be the intermediate node pair?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 6
template['source_argv'] = ['node_id','node_id_list','node_id_tuple_list','node_id','node_id','node_id_pair']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "3-1-3-3"

task_subgroup_3["3-1-3-3"] = template

template = {}

template['source']="{} is connected with {} within three hops through {}, respectively. Now I want {} to be connected to {} in three hops, which intermediate node pair should I choose?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 5
template['source_argv'] = ['node_id', 'node_id_list','node_id_tuple_list','node_id','node_id']
template['target_argc'] = 1
template['target_argv'] = ['node_id_pair']
template['id'] = "3-1-3-4"

task_subgroup_3["3-1-3-4"] = template


#



template = {}

template['source']="{} is connected with {} within one hop through featured edges: {}, respectively. Now I want {} to be connected to {} in two hops through featured path: {}, should {} be the intermediate node?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 7
template['source_argv'] = ['node_id', 'node_id_list','edge_feature_list','node_id','node_id','feature_tuple','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "3-2-1-1"

task_subgroup_3["3-2-1-1"] = template

template = {}

template['source']="{} is connected with {} within one hop through featured edges: {}, respectively. Now I want {} to be connected to {} in two hops through featured path: {}, which intermediate node should I choose?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 6
template['source_argv'] = ['node_id', 'node_id_list','edge_feature_list','node_id','node_id','feature_tuple']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "3-2-1-2"

task_subgroup_3["3-2-1-2"] = template

template = {}

template['source']="{} linked to nodes within one hop in the following order: {} through the following featured edges: {}. Now I want {} to be connected to {} in two hops through featured path: {}, should {} be the intermediate node?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 7
template['source_argv'] = ['node_id', 'node_id_list','edge_feature_list','node_id','node_id','feature_tuple','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "3-2-1-3"

task_subgroup_3["3-2-1-3"] = template

template = {}

template['source']="{} linked to nodes within one hop in the following order: {} through the following featured edges: {}. Now I want {} to be connected to {} in two hops through featured path: {}, which intermediate node should I choose?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 6
template['source_argv'] = ['node_id', 'node_id_list','edge_feature_list','node_id','node_id','feature_tuple']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "3-2-1-4"

task_subgroup_3["3-2-1-4"] = template

template = {}

template['source']="{} is connected with {} within two hops through featured paths: {}, respectively. Now I want {} to be connected to {} in two hops through featured path: {}, should {} be the intermediate node?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 7
template['source_argv'] = ['node_id', 'node_id_list','feature_tuple_list','node_id','node_id','feature_tuple','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "3-2-2-1"

task_subgroup_3["3-2-2-1"] = template

template = {}

template['source']="{} is connected with {} within two hops through featured paths: {}, respectively. Now I want {} to be connected to {} in two hops through featured path: {}, which intermediate node should I choose?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 6
template['source_argv'] = ['node_id', 'node_id_list','feature_tuple_list','node_id','node_id','feature_tuple']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "3-2-2-2"

task_subgroup_3["3-2-2-2"] = template

template = {}

template['source']="{} is connected with {} within two hops through {} and featured paths: {}, respectively. Now I want {} to be connected to {} in two hops through featured path: {}, should {} be the intermediate node?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 8
template['source_argv'] = ['node_id','node_id_list','node_id_list','feature_tuple_list','node_id','node_id','feature_tuple','node_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "3-2-2-3"

task_subgroup_3["3-2-2-3"] = template

template = {}

template['source']="{} is connected with {} within two hops through {} and featured paths: {}, respectively. Now I want {} to be connected to {} in two hops through featured path: {}, which intermediate node should I choose?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 7
template['source_argv'] = ['node_id','node_id_list','node_id_list','feature_tuple_list','node_id','node_id','feature_tuple']
template['target_argc'] = 1
template['target_argv'] = ['node_id']
template['id'] = "3-2-2-4"

task_subgroup_3["3-2-2-4"] = template


#
template = {}

template['source']="{} is connected with {} within three hops through featured paths: {}, respectively. Now I want {} to be connected to {} in three hops through featured path: {}, should {} be the intermediate node pair?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 7
template['source_argv'] = ['node_id', 'node_id_list','feature_triple_list','node_id','node_id','feature_triple','node_id_pair']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "3-2-3-1"

task_subgroup_3["3-2-3-1"] = template

template = {}

template['source']="{} is connected with {} within three hops through featured paths: {}, respectively. Now I want {} to be connected to {} in three hops through featured path: {}, which intermediate node pair should I choose?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 6
template['source_argv'] = ['node_id', 'node_id_list','feature_triple_list','node_id','node_id','feature_triple']
template['target_argc'] = 1
template['target_argv'] = ['node_id_pair']
template['id'] = "3-2-3-2"

task_subgroup_3["3-2-3-2"] = template

template = {}

template['source']="{} is connected with {} within three hops through {} and featured paths: {}, respectively. Now I want {} to be connected to {} in three hops through featured path: {}, should {} be the intermediate node pair?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 8
template['source_argv'] = ['node_id','node_id_list','node_id_tuple_list','feature_triple_list','node_id','node_id','feature_triple','node_id_pair']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "3-2-3-3"

task_subgroup_3["3-2-3-3"] = template

template = {}

template['source']="{} is connected with {} within three hops through {} and featured paths: {}, respectively. Now I want {} to be connected to {} in three hops through featured path: {}, which intermediate node pair should I choose?"
template['target'] = "{}"
template['task'] = 'intermediate'
template['source_argc'] = 7
template['source_argv'] = ['node_id','node_id_list','node_id_tuple_list','feature_triple_list','node_id','node_id','feature_triple']
template['target_argc'] = 1
template['target_argv'] = ['node_id_pair']
template['id'] = "3-2-3-4"

task_subgroup_3["3-2-3-4"] = template

all_tasks['intermediate'] =  task_subgroup_3
