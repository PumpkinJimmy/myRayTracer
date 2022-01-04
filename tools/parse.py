import string
TOKEN_NULL = 0
TOKEN_NUM = 1
TOKEN_STRING = 2
TOKEN_LIST_BEGIN = 3
TOKEN_LIST_END = 4
TOKEN_BLOCK_BEGIN = 5
TOKEN_BLOCK_END = 6
TOKEN_KEYWORD = 7
def get_token(s, p):
    slen = len(s)
    while p < slen and s[p] in string.whitespace:
        p += 1
    if p >= slen:
        return (TOKEN_NULL, p, None)
    if s[p] == '"':
        p2 = s.find('"', p+1)
        if p2 == -1:
            raise Exception(f"col {p}: expect \" got nothing")
        return (TOKEN_STRING, p2 + 1, s[p+1:p2])
    if s[p] == '[':
        return (TOKEN_LIST_BEGIN, p+1, None)
    if s[p] == ']':
        return (TOKEN_LIST_END, p+1, None)
    p2 = p
    while p2 < slen and s[p2] not in string.whitespace:
        p2 += 1
    if s[p:p2].endswith('Begin'):
        return (TOKEN_BLOCK_BEGIN, p2, s[p:p2-5], None)
    if s[p:p2].endswith('End'):
        return (TOKEN_BLOCK_END, p2, s[p:p2-3], None)
    else:
        p2 = p
        while p2 < slen and s[p2] in '0123456789.e-':
            p2 += 1
        try:
            return (TOKEN_NUM, p2, eval(s[p:p2]))
        except:
            p2 = p
            while p2 < slen and s[p2] not in string.whitespace:
                p2 += 1
            return (TOKEN_KEYWORD, p2, s[p:p2])
def parse_args(s):
    p = 0
    slen = len(s)
    res = []
    stack = []
    stack_data = []
    while p < slen:
        (t, p, data) = get_token(s,p)
        if t == TOKEN_NULL:
            return res
        elif t == TOKEN_LIST_BEGIN:
            stack.append('[')
            stack_data.append(res)
            res = []
        elif t == TOKEN_BLOCK_BEGIN:
            stack.append(data)
            stack_data.append(res)
            res = []
        elif t == TOKEN_LIST_END:
            stack.pop()
            tmp = res
            res = stack_data.pop()
            res.append(tmp)
        elif t == TOKEN_BLOCK_END:
            name= stack.pop()
            tmp = res
            res = stack_data.pop()
            res.append((name, tmp))
        else:
            res.append(data)
    return res
def parse(s):
    res = []
    res_stack = []
    lines = s.strip().split('\n')
    k = 'Scene'
    for line in lines:
        tks = line.strip().split(' ', 1)
        if len(tks)>1:
            res.append({
                'key': tks[0],
                'args': parse_args(tks[1])
            })
        else:
            tk = tks[0]
            if tk.endswith('Begin'):
                res_stack.append((k, res))
                res = []
                k = tk[:-5]
            else:
                sub_k = k
                sub_res = res
                (k, res) = res_stack.pop()
                res.append({
                    'key': k,
                    'commands': sub_res
                })
    return res