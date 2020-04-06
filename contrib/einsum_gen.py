import numpy as np
import numba as nb

indend= " " *4

def parse_einsum_str(einsum_str):
    lhs,rhs=einsum_str.split("->")
    lhs=lhs.split(",")
    
    charset=set()
    for lhs_ in lhs:
        for s in lhs_:
            charset.add(s)
    return charset, lhs,rhs

def write_header(s,arr_names):
    s+="def einsum("
    for i in range(len(arr_names)-1):
        s+=arr_names[i]+","
    s+=arr_names[-1]+"):\n"
    return s

def write_size_assert(s,ind_level,sizes,asserts):
    ind_level+=1
    for i in range(len(sizes)):
        s+= indend *ind_level + sizes[i]+"\n"
    
    s+= indend *ind_level +"\n"
    for i in range(len(asserts)):
        s+= indend *ind_level + asserts[i]+"\n"
    
    s+= " " * ind_level*4 +"\n"
    return s,ind_level

def get_size(char_list_lhs,arr_names,arr):
    sizes=[]
    asserts=[]
    dict_sizes={}
    
    charset=set()
    for i ,chars in enumerate(char_list_lhs):
        for j,char in enumerate(chars):
            if char not in charset:
                sizes.append(char+"_s = " + arr_names[i] + ".shape["+str(j)+"]")
                dict_sizes[char] = arr[i].shape[j]
                charset.add(char)
            else:
                asserts.append("assert " + char+"_s == " + arr_names[i] + ".shape["+str(j)+"]")
    
    return sizes, asserts ,dict_sizes

def get_loop_order(lhs,rhs,charset,dict_sizes,idx_rm):
    cost = dict.fromkeys(charset, 0)
    
    for i ,lhs_ in enumerate(lhs):
        #C-order assumed
        stride=1
        for j in range(len(lhs_)-2,-1,-1):
            stride*=dict_sizes[lhs_[j+1]]
            cost[lhs_[j]]+=stride
    
    if len(rhs)>0:
        stride=1
        for j in range(len(rhs)-2,-1,-1):
            stride*=dict_sizes[rhs[j+1]]
            cost[rhs[j]]+=stride
    
    is_simple=False
    
    #cost model
    cost_model=[(k, v) for k, v in sorted(cost.items(), key=lambda item: item[1],reverse=True)]
    
    return cost_model,is_simple

def gen_op(inds,char_list_lhs,arr_names):
    op="+="
    for i in range(len(inds)-1):
        op+=arr_names[inds[i]] + gen_ind(char_list_lhs[i]) + " *"
    op+=arr_names[inds[-1]] + gen_ind(char_list_lhs[-1])
    return op

def gen_ind(chars,suffix=""):
    s=""
    if len(chars)==0:
        return s
    s+="["
    for i in range(len(chars)-1):
        s+=chars[i] + suffix +","
    s+=chars[-1] + suffix + "]"
    return s

def gen_transpose_inds(inds):
    s=""
    for i in range(len(inds)-1):
        s+=str(inds[i]) +","
    s+=str(inds[-1])
    return s

def gen_reshape_ind(chars_list,suffix=""):
    s=""
    for ii in range(len(chars_list)-1):
        chars=chars_list[ii]
        for i in range(len(chars)-1):
            s+=chars[i] + suffix +"*"
        s+=chars[-1] + suffix + ","
    
    chars=chars_list[-1]
    for i in range(len(chars)-1):
        s+=chars[i] + suffix +"*"
    s+=chars[-1] + suffix
    return s

def write_tensordot(s,ind_level,red,res_name,num_Ten):
    dot_name=[]
    for i in range(2):
        if len(red[i]["permute"])==0:
            s+=indend* ind_level + "Ten_"+str(num_Ten)+ " = " + red[i]["name"] + \
                ".reshape("+ gen_reshape_ind(red[i]["reshape"],suffix="_s") +")"
            
            dot_name.append("Ten_"+str(num_Ten))
            num_Ten+=1
        else:
            s+=indend* ind_level + "Ten_"+str(num_Ten)+ " = " + "np.ascontiguousarray(" + \
                 red[i]["name"] + ".transpose(" +  gen_transpose_inds(red[i]["permute"]) + "))\n"

            s+=indend* ind_level + "Ten_"+str(num_Ten+1)+ " = " + "Ten_"+str(num_Ten) + \
                ".reshape("+ gen_reshape_ind(red[i]["reshape"],suffix="_s") +")"
            
            dot_name.append("Ten_"+str(num_Ten+1))
            num_Ten+=2
        if red[i]["transpose"]:
            s+=".T\n"
        else:
            s+="\n"
    
    #check if transpose is needed
    if np.array_equal(red[-1][0], np.arange(len(red[-1][0]))):
        s+=indend* ind_level+ res_name + " = np.dot(" + dot_name[0] + "," + dot_name[1] +")" + \
            ".reshape(" + gen_reshape_ind(red[-1][1],suffix="_s") +")\n"
    else:
        s+=indend* ind_level+"Ten_"+str(num_Ten) + " = np.dot(" + dot_name[0] + "," + dot_name[1] +")" + \
            ".reshape(" + gen_reshape_ind(red[-1][1],suffix="_s") +")\n"

        s+=indend* ind_level + res_name + " = np.ascontiguousarray(" + "Ten_"+str(num_Ten) + ".transpose(" + \
            gen_transpose_inds(red[-1][0]) +"))\n"
        num_Ten+=1
    
    s+=indend* ind_level + "\n"
    return s, num_Ten

def containsAll(str, set):
    return 0 not in [c in str for c in set]

def tensordot(lhs,rhs,arr_names,inds,idx_rm,s0,s1):
    idx_rm_str="".join(idx_rm)
    inds=list(inds)
    red=[]

    #optimize idx_rm
    if s0 < s1: 
        lhs.reverse()
        inds.reverse()

    left_arr_1=lhs[0][:len(idx_rm)]
    right_arr_1=lhs[0][-len(idx_rm):]
    left_arr_2=lhs[1][:len(idx_rm)]
    right_arr_2=lhs[1][-len(idx_rm):]
    if containsAll(left_arr_1, idx_rm):    idx_rm_str = left_arr_1
    elif containsAll(right_arr_1, idx_rm): idx_rm_str = right_arr_1
    elif containsAll(left_arr_2, idx_rm):  idx_rm_str = left_arr_2
    elif containsAll(right_arr_2, idx_rm): idx_rm_str = right_arr_2

    info=[]
    for lhs_ in lhs:
        contract=[]
        remainder=[]
        remainder_str=""
        for i,s in enumerate(idx_rm_str):
            contract.append(lhs_.find(s))
        for i,s in enumerate(lhs_):
            if s not in idx_rm_str:
                remainder.append(i)
                remainder_str+=s
        info.append((contract,remainder,remainder_str))

    #first array
    if right_arr_1==idx_rm_str:
        red.append({"name":arr_names[inds[0]],"transpose":False,
            "reshape":((lhs[0][:-len(idx_rm)]),(right_arr_1)),"permute":(),"remainder_str":info[0][2]})
    elif left_arr_1==idx_rm_str:
        red.append({"name":arr_names[inds[0]],"transpose":True,
            "reshape":((left_arr_1),(lhs[0][len(idx_rm):])),"permute":(),"remainder_str":info[0][2]})
    else:
        permute=tuple(info[0][1]+info[0][0])
        reshape=(info[0][2],idx_rm_str)
        red.append({"name":arr_names[inds[0]],"transpose":False,
            "reshape":reshape,"permute":permute,"remainder_str":info[0][2]})

    #second array   
    if right_arr_2==idx_rm_str:
        red.append({"name":arr_names[inds[1]],"transpose":True,
            "reshape":((lhs[1][:-len(idx_rm)]),(right_arr_2)),"permute":(),"remainder_str":info[1][2]})

    elif left_arr_2==idx_rm_str:
        red.append({"name":arr_names[inds[1]],"transpose":False,
            "reshape":((left_arr_2),(lhs[1][len(idx_rm):])),"permute":(),"remainder_str":info[1][2]})
    else:
        permute=tuple((info[1][1]+info[1][0]))
        reshape=((info[1][2],idx_rm_str))
        red.append({"name":arr_names[inds[1]],"transpose":True,
            "reshape":reshape,"permute":permute,"remainder_str":info[1][2]})
    
    #Transpose the operation if it fits better to the result
    if rhs.find(red[1]["remainder_str"])==0 or \
        rhs.find(red[0]["remainder_str"])==len(rhs)-len(red[0]["remainder_str"]):
        red.reverse()
        red[0]["transpose"]= not red[0]["transpose"]
        red[1]["transpose"]= not red[1]["transpose"]
    
    remainder=(red[0]["remainder_str"]+red[1]["remainder_str"])
    remainder_inds=[remainder.find(s) for s in rhs]
    red.append((remainder_inds,tuple([s for s in remainder])))
    return red

def write_reduction(s,ind_level,num_TMP,arr_names,contraction,operands_str,dict_sizes,num_Ten):
    inds, idx_rm, einsum_str, remaining, blas = contraction
    charset, lhs,rhs=parse_einsum_str(einsum_str)
    
    #get reduction size
    s_inner=1
    for char in idx_rm: s_inner*=dict_sizes[char]
    
    s0=1
    s1=1
    sout1=1
    sout2=1    
    
    if len(lhs)==2:
        # Get array sizes 
        for char in lhs[0]: s0*=dict_sizes[char]
        for char in lhs[1]: s1*=dict_sizes[char]

        #Get outer sizes
        for char in lhs[0]: 
            if not char in idx_rm: sout1*=dict_sizes[char]
        for char in lhs[1]: 
            if not char in idx_rm: sout2*=dict_sizes[char]
    
    res_name="TMP_"+str(num_TMP)
    num_TMP+=1
    arr_names.append(res_name)
    
    #BLAS call is not considered beneficial if reduction size and outer sizes are bellow 10
    if blas and len(rhs) >0 and (sout1>=10 or sout2 >=10 or s_inner >=10)
        red=tensordot(lhs,rhs,arr_names,inds,idx_rm,s0,s1)
        s,num_Ten=write_tensordot(s,ind_level,red,res_name,num_Ten)
    else:
        #find optimal loop order (minimal stride between memory access)
        cost,is_simple=get_loop_order(lhs,rhs,charset,dict_sizes,idx_rm)
        
        if len(rhs)==0:
            s+=indend *ind_level+ res_name +" = 0 \n"
            s+=indend *ind_level + "for " + cost[0][0] + " in nb.prange(" + cost[0][0] +"_s):\n"
            ind_level+=1
            for i in range(1,len(cost)):
                s+=indend *ind_level + "for " + cost[i][0] + " in range(" + cost[i][0] +"_s):\n"
                ind_level+=1
            s+=indend *ind_level + arr_names[-1] + gen_op(inds,lhs,arr_names) +"\n"
        else:
            #allocate output/temporary array
            size_str=""
            for char in rhs: 
                size_str+=char+"_s,"

            s+=indend *ind_level+ res_name +" = np.zeros(("+ size_str[:-1]+"))"+"\n"

            if cost[0][0] in idx_rm:
                #parfor not possible (at least without changing loop order)
                s+=indend *ind_level + "for " + cost[0][0] + " in range(" + cost[0][0] +"_s):\n"
                ind_level+=1
            else:
                s+=indend *ind_level + "for " + cost[0][0] + " in nb.prange(" + cost[0][0] +"_s):\n"
                ind_level+=1
            
            #Write reduction
            for i in range(1,len(cost)):
                s+=indend *ind_level + "for " + cost[i][0] + " in range(" + cost[i][0] +"_s):\n"
                ind_level+=1

            s+=indend *ind_level + arr_names[-1] + gen_ind(rhs) + gen_op(inds,lhs,arr_names) +"\n"

        ind_level-=len(cost)
        s+=indend *ind_level + "\n"
    

    for ind in sorted(inds,reverse=True):
        del arr_names[ind]
    
    return s,ind_level,num_TMP,arr_names,num_Ten

def write_add_asserts(add_asserts_str,dict_sizes,s,ind_level):
    for char in add_asserts_str:
        s+=ind_level*indend + "assert " + char +"_s == " + str(dict_sizes[char]) +"\n"
    s+=ind_level*indend + "\n"
    
    return s


def einsum_gen(operands_str,*arr,**kwargs):
    """
    Input parameters:
    
    - operands_str 
      
      eg. 'oij,imj,mjkn,lnk,plk->op' 
      * ellipsis are not supported
      * It must be a reduction
    
    - arr
      The arrays to apply the function (always have to be C-contiguous)
      
      Important: 
      * For finding the optimal reduction path np.einsum(...,optimize='optimal') is used.
      * The arrays should have a relastic size for optimal code generation.
      
    - kwargs (optional)
      * arr_names (list of strings) eg. ["A","B","C"]
      If the input arrays should have a special name this can be set here 
      Ten_ and TMP are not allowed because this array names are used internally
      
      * add_asserts_str (string with indices) eg. eg. "ijs"
      Dimensions specified here are fixed with an assert statement.
      This can be beneficial on very small dims which are always the same, because
      it is often possible to get an additional speedup if LLVM is able to unroll loops
    
    Output:
    String of the generated function.
    """
    
    
    num_TMP=0
    ind_level=0
    num_Ten=0
    s=""
    
    arr_names=[]
    add_asserts_str=""
    for key, value in kwargs.items():
        if key=="arr_names":
            arr_names=value
        if key=="add_asserts_str":
            add_asserts_str=value
    
    if len(arr_names)==0:
        for i in range(len(arr)):
            arr_names.append("arg_"+str(i))
    
    lhs,rhs=operands_str.split("->")
    lhs=lhs.split(",")
    
    #call einsum path
    operands, contraction_list=np.einsum_path(operands_str,*arr,optimize='optimal',einsum_call=True)
    
    #get sizes and asserts
    charset, char_list_lhs,char_list_rhs=parse_einsum_str(operands_str)
    sizes_str, asserts_str, dict_sizes = get_size(lhs,arr_names,arr)
    
    #generate Header
    s=write_header(s,arr_names)
    s,ind_level=write_size_assert(s,ind_level,sizes_str,asserts_str)
    
    #generate additional asserts
    if len(add_asserts_str)>0:
        s=write_add_asserts(add_asserts_str,dict_sizes,s,ind_level)
    
    #generating contractions
    for num, contraction in enumerate(contraction_list):
        s,ind_level,num_TMP,operands_str,num_Ten=write_reduction(s,ind_level,
                num_TMP,arr_names,contraction,operands_str,dict_sizes,num_Ten)
    
    s+=indend *ind_level + "return "+ arr_names[-1] +"\n"
        
        
    return s
