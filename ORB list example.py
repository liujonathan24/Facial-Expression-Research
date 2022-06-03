


input_list = [40, 48, 123, 88, 109, 213, 217, 88, 203, 76, 180, 40, 26, 222, 182, 48, 204, 177,
     215, 0, 129, 35, 186, 208, 83, 217, 33, 187, 83, 8, 102, 247]

def int_to_binary(integer):
    result = []
    for i in range(8):
        result.append((integer >> i) % 2)
    result.reverse()
    return result

def byte_list_to_binary(input_list):
    final_list = [digit for a in input_list for digit in int_to_binary(a) ]
    return final_list



