# For every template(i.e. instruction prompt), the necessary attributes are: ['source','target','task','id']


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

template['source']="{} is connected with {} within three hops through {}, respectively. Will {} be connected to {} within three hops through {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 6
template['source_argv'] = ['node_id','node_id_list','node_id_tuple_list','node_id','node_id','node_id_tuple']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-1-3-3"

task_subgroup_1["1-1-3-3"] = template




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

template['source']="{} is connected with {} within three hops through {} and featured paths: {}, respectively. Will {} be connected to {} within three hops through {} and featured path: {}?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 8
template['source_argv'] = ['node_id','node_id_list','node_id_tuple_list','feature_triple_list','node_id','node_id','node_id_tuple','feature_triple']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-2-3-3"

task_subgroup_1["1-2-3-3"] = template


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

template['source']="({},{}) is connected with {} within three hops through {}, respectively. Will ({},{}) be connected to ({},{}) within three hops through ({},{}; {},{})?"
template['target'] = "{}"
template['task'] = "link"
template['source_argc'] = 9
template['source_argv'] = ['node_id','node_id_list','node_id_tuple_list','node_id','node_id','node_id_tuple']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-3-3-3"

task_subgroup_1["1-3-3-3"] = template






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

template={}

template['source']=" Node represents academic paper with a specific topic. node ({},{}) is featured with its abstract: {}. Which category should ({},{}) be classified as?"
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



