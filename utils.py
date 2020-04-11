def get_table_row(items):
    row = ''
    for item in items:
        row += '| {:s} '.format(item)
    row += '|\n'
    return row

def generate_md_table(item_list, title=None, col_num=4):
    '''
    args: 
        item_list: a list of items (str or can be converted to str)
        that want to be presented in table.

        title: None, or a list of strings. When set to None, empty title
        row is used and column number is determined by col_num; Otherwise, 
        it will be used as title row, its length will override col_num.

    return: 
        table: markdown table string.
    '''
    table = ''
    if title is not None:
        col_num = len(title)
        table += get_table_row(title)
    else:
        table += get_table_row([' ']*col_num) # empty title row
    table += get_table_row(['-'] * col_num)
    for i in range(0, len(item_list), col_num):
        table += get_table_row(item_list[i:i+col_num])
    return table