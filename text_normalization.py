# -*- coding: utf-8 -*-
import nltk
from pyvi.pyvi import ViTokenizer
import re
import ast


with open('text/acronym.txt', 'r') as f:
   acronym = ast.literal_eval(str(f.read()))

def int_to_vn(num):
    d = {0: 'không', 1: 'một', 2: 'hai', 3: 'ba', 4: 'bốn', 5: 'năm', 6: 'sáu', 7: 'bảy', 8: 'tám', 9: 'chín', 10: 'mười'}
    if num <= 10: return d[num]
    if num//1000000000 > 0:
        if num % 1000000000 == 0: return int_to_vn(num // 1000000000) + " tỷ"
        if num%1000000000 < 10:
            return int_to_vn(num//1000000000) + " tỷ không triệu không nghìn không trăm linh "+int_to_vn(num % 1000000000)
        if num % 1000000000 < 100:
            return int_to_vn(num // 1000000000) + " tỷ không triệu không nghìn không trăm " + int_to_vn(num % 1000000000)
        if num % 1000000000 < 1000:
            return int_to_vn(num // 1000000000) + " tỷ không triệu không nghìn " + int_to_vn(num % 1000000000)
        if num % 1000000000 < 1000000:
            return int_to_vn(num // 1000000000) + " tỷ không triệu " + int_to_vn(num % 1000000000)
        if num % 1000000000 != 0:
            return int_to_vn(num // 1000000000) + " tỷ " + int_to_vn(num % 1000000000)
    if num//1000000 > 0:
        if num % 1000000 == 0: return int_to_vn(num // 1000000) + " triệu"
        if num%1000000 < 10:
            return int_to_vn(num//1000000) + " triệu không nghìn không trăm linh "+int_to_vn(num % 1000000)
        if num % 1000000 < 100:
            return int_to_vn(num // 1000000) + " triệu không nghìn không trăm " + int_to_vn(num % 1000000)
        if num % 1000000 < 1000:
            return int_to_vn(num // 1000000) + " triệu không nghìn " + int_to_vn(num % 1000000)
        if num % 1000000 != 0:
            return int_to_vn(num // 1000000) + " triệu " + int_to_vn(num % 1000000)
    if num // 1000 > 0:
        if num % 1000 == 0: return int_to_vn(num//1000) + " nghìn"
        if num%1000 <10:
            return int_to_vn(num//1000) + " nghìn không trăm linh "+int_to_vn(num%1000)
        if num%1000 <100:
            return int_to_vn(num//1000) + " nghìn không trăm "+int_to_vn(num%1000)
        if num%1000 != 0:
            return int_to_vn(num//1000) + " nghìn "+int_to_vn(num%1000)
    if num // 100 > 0:
        if num%100 == 0:
            return int_to_vn(num // 100) + " trăm"
        if num%100 <10:
            return int_to_vn(num//100) + " trăm linh " + int_to_vn(num%100)
        if num%100 != 0:
            return int_to_vn(num//100) +  " trăm " + int_to_vn(num%100)
    if num // 10 > 0 and num >= 20:
        if num%10 != 0:
            if num%10 == 5:
                return int_to_vn(num//10) + ' mươi lăm'
            if num%10 == 1:
                return int_to_vn(num//10) + ' mươi mốt'
            if num%10 == 4:
                return int_to_vn(num//10) + ' mươi tư'
            return int_to_vn(num // 10) + ' mươi ' + int_to_vn(num % 10)
        return int_to_vn(num//10) + ' mươi'
    if num // 10 > 0:
        if num == 15:
            return 'mười lăm'
        return "mười "+ d[num%10]
def _hour(m):
    out = m.group(0)
    out = out.replace(',', '').replace('.', '').strip().split(':')
    check = False
    if len(out) == 3:
        if int(out[2].strip()) > 0:
            check = True
        out = out[0] + ' giờ ' + out[1] + ' phút ' + out[2] + ' giây '
    else:
        out = m.group(0)
    out = out.replace(':', ' giờ ')
    out = out.replace('00 giờ', '0 giờ')
    if check is not True:
        out = out.replace('00 phút', '')
        out = out.replace('00 giây', '')
        out = out.replace('00', '')
    return out
def _size(m):
    return m.group(0).replace('x', ' nhân ')
def _old(m):
    return m.group(0).replace('u', 'u ')
def _g(m):
    print('output ', m.group(0))
    return m.group(0).replace('g', ' giờ ')
def _hour_minute(m):
    out = m.group(0)
    end  = ''
    print('step 2 ', out)
    if out[-1]==':':
        end = ', '
    elif out[-1]=='-':
        end = ' đến '
    elif out[-1]=='–':
        end = ' đến '
    out = out[:-1].strip()
    out = out.replace('h:', ' giờ ')
    out = out.replace('g:', ' giờ ')
    out = re.sub(re.compile(r"[0-9]g"), _g, out)
    out = out.replace('h', ' giờ ')
    out = out.replace(':', ' giờ ')

    if out[-1].isdigit():
        out = out + ' phút '
    elif out[-1] == 'p':
        out = out[:-1] + ' phút '
    out = out.replace('00 giờ', '0 giờ')
    out = out.replace('00 phút', '')
    out = out.replace('00', '')
    out = out.replace(' 0', ' ')
    out = out.replace('-', ' đến ')
    out = out.replace('–', ' đến ')
    return out + end
def _hour_minute1(m):
    print('input ', m.group(0))
    output = re.sub(re.compile(r"([0-9]|[0-2][0-9])((\ g\ )|(\ g)|(\ g)|(\ g:\ )|(\ g:)|(g:\ )|(g:)|g)((([0-9]|[0-6][0-9])(p|))|)(:|\.|,|-| -|–| –|\ )"), _hour_minute, m.group(0))
    return output
def _bignum(m):
    return m.group(0).replace('.', '')
def _float(m):
    x = m.group(0).replace('.', '')
    x = x.split(',')
    output = x[0] + ' phẩy '
    if len(x[1]) > 2:
        output += ' '.join(list(x[1]))
    else: output += x[1]
    return output
def _thounsand(m):
    return m.group(0).replace('k', ' nghìn').replace('-', ' đến ')
def _m(m):
    return m.group(0).replace('m', ' mét ')
def _v(m):
    return m.group(0).replace('v', ' vol')
def _weight(m):
  return m.group(0).replace('g', ' gam ')
def _volume(m):
  return m.group(0).replace('l', ' lít ')
def _ward(m):
    return m.group(0).replace('.', ' ').replace(' p ', ' phường ')
def _district(m):
    return m.group(0).replace('.', ' ').replace(' q ', ' quận ')
def _city(m):
    return m.group(0).replace(' tp.', ' thành phố ').replace(' t.p', ' thành phố ').replace(' tx.', ' thị xã ').replace(' tt.', ' thị trấn ')
def _money(m):
    return m.group(0).replace('đ', ' đồng ')
def _split(m):
    out = ''
    text = m.group(0)
    for idx, char in enumerate(text):
        if not char.isdigit():
            out += char
        else:
            if char == '0' or len(text[idx:])>2:
                out += ' ' + ' '.join(list(text[idx:]))
            else:
                out += ' ' + text[idx:]
            break
    return out
def _split2(m): #OnePlus Nord N10 D12345
    out1, out2 = '', ''
    text = m.group(0).strip()
    for idx, char in enumerate(text):
        if not char.isdigit():
            out1 += char 
        else:
            if char == '0' or len(text[idx:])>2:
                out2 += ' ' + ' '.join(list(text[idx:])) #ABC 0123
            else:
                out2 += ' ' + text[idx:] #ABC 1 2 3 
            break
    if out1 not in ['VOV', 'VOH', 'VTV', 'HTV']:
        out1 = ' '.join(list(out1))
    return ' '+out1+' '+out2+' '
def _split3(m):
    result = ''
    for w in list(m.group(0).strip()):
        if w.isdigit():
            result += ' ' + int_to_vn(int(w))
        else:  result += ' ' + w
    return result + ' ' 
def _split4(m):
    text = m.group()
    for i in text:
        if not i.isdigit() and i != ' ':
            text = text.replace(i, ' ' + i)
            break
    return text
def _uper(m):
    return m.group(0).replace(m.group(0), ' '.join(list(m.group(0))))
def _phone(m):
    out = m.group(0).replace('.', '')
    return ' '.join(list(out))
def _phone2(m):
    out = m.group(0).replace('.', '').strip()
    for x in out.split(' '):
        if x.isdigit():
            out = out.replace(x, '  '.join(list(x)))
    return ' '+out+' '
def _no(m):
    out = m.group(0).split('.')
    return out[0]+' '+int_to_vn(int(out[1]))
def _num(m):
    text = str(m.group(0)).split(' ')
    result = ''
    for id, x in enumerate(text):
        if x.isdigit():
            if id > 0 and text[id-1] in ['thứ', 'tháng'] and int(x)==4:
                result += 'tư '
            else:
                result+= str(int(x)) + ' '
        else: result+=x+' '
    return result
def _no(m):
    out = m.group(0).split('.')
    return out[0]+' '+int_to_vn(int(out[1]))
def _link(m):
    out = m.group(0)
    out = out.replace('/', ' siệt ' )
    out = out.replace('.', ' chấm ')
    out = out.replace(':', ' hai chấm ')
    out = out.replace('-', ' gạch giữa ')
    out = out.replace('vn', 'v n')
    out = out.replace('org', 'o r g')
    return out
def _mail(m):
    out = m.group(0)
    out = out.replace('/', ' siệt ')
    out = out.replace('.', ' chấm ')
    out = out.replace('@', ', a còng, ')
    out = out.replace(':', ' hai chấm ')
    out = out.replace('olli-ai', 'olli ây ai')
    out = out.replace('gmail', 'gờ mail')
    return out
def _license_plate(m):
    out = ''
    for char in m.group(0):
        if char.isdigit():
            out+=char+' '
        else: out+=char
    return out
def _roman_num(word):
    num = 0
    p = False
    for x in list(word)[::-1]:
        if x == 'i':
            if p:
                p = False
                num -= 1
            else: num += 1
        elif x == 'v':
            num += 5
            p = True
        elif x == 'x':
            num += 10
            p = True
    return str(num)

def _roman_numerals(m):
    out = ''
    #print(m.group(0))
    compiles = re.compile(r'(x|i|v)+')
    for w in nltk.word_tokenize(m.group(0)):
        if compiles.match(w) is not None:
            out += _roman_num(w) + ' '
        else:
            out += w + ' '
    return out
def _dot(m):
    return m.group(0).replace('.', '')

def _dot2(m):
    return m.group(0).replace('.', ' ')
def _dot3(m):
    x = m.group(0).split('.')
    output = x[0] + ' chấm ' + ' '.join(list(x[1]))
    return output
def _dot4(m):
    return m.group(0).replace('.', ' chấm ')
def _measure(m):
    input = m.group(0)
    input = input.replace('km2', ' ki lô mét vuông')
    input = input.replace('m2', ' mét vuông')
    input = input.replace('m3/s', ' mét khối trên giây')
    input = input.replace('m3', ' mét khối')
    input = input.replace('km/h', ' ki lô mét trên giờ')
    input = input.replace('m/s', ' mét trên giây')
    input = input.replace('°c', ' độ xê')
    input = input.replace('°f', ' độ ép')
    input = input.replace('ml', ' mi li lít')
    input = input.replace('mg', ' mi li gam')
    input = input.replace('cm', ' xen ti mét')
    input = input.replace('nm', ' na nô mét')
    input = input.replace('mm', ' mi li mét')
    input = input.replace('ms', ' mi li giây')
    input = input.replace('m³', ' mét khối')
    input = input.replace('mw', ' mê ga oát')
    input = input.replace('kwh', ' ki lô oát giờ')
    input = input.replace('km²', ' ki lô mét vuông')
    input = input.replace('km', ' ki lô mét')
    input = input.replace('đ/kg', 'đồng trên kí')
    input = input.replace('đồng/kg', 'đồng trên kí')
    input = input.replace('đồng/km', 'đồng trên kí lô mét')
    input = input.replace('kg', ' ki lô gam')
    input = input.replace('kw', ' ki lô oát')
    input = input.replace('độ c', 'độ xê')
    input = input.replace('$', ' đô_la')
    input = input.replace('%', ' phần_trăm')
    input = input.replace('m²', ' mét vuông')
    input = input.replace('mhz', ' mê ga hét')
    input = input.replace('khz', ' ki lô hét')
    input = input.replace('hz', ' hẹt')
    input = input.replace('gb', ' ghi ga bai')
    input = input.replace('µm', ' mi rô mét')
    input = input.replace('ft', ' feet')
    input = input.replace('mmhg', ' mi li mét thủy ngân')
    input = input.replace('ha', ' héc ta')
    input = input.replace('mah', ' mi li am pe giờ')
    input = input.replace('vnđ', ' việt_nam_đồng')
    input = input.replace('vnd', ' việt_nam_đồng')
    input = input.replace('ndt', ' nhân_dân_tệ')
    input = input.replace('€', ' ơ_rô')
    input = input.replace('£', 'bản_anh')
    return input
def _interval(m):
    out = m.group(0).replace('-', ' đến ngày ')
    return out
def _ddmmyy(m):
    text = m.group(0)
    out = ''
    if len(text.split('/'))==3:
        date = text.split('/')
    elif len(text.split('-'))==3:
        date = text.split('-')
    elif len(text.split('.'))==3:
        date = text.split('.')
    if int(date[1]) == 4:
        out = int_to_vn(int(date[0])) + ' tháng tư năm '+ int_to_vn(int(date[2]))
    else:
        out = int_to_vn(int(date[0])) + ' tháng ' + int_to_vn(int(date[1])) + ' năm ' + int_to_vn(int(date[2]))
    return out + ' '

def _mmyy(m):
    text = m.group(0).strip()
    end = ''
    if text[-1] in ',.?!':
        en = text[-1]
        text = text[:-1]
    out = ''
    if len(text.split(' '))>1:
        date = text.split(' ')[1]
    else: date = text
    if len(date.split('/'))==2:
        date = date.split('/')
    elif len(date.split('-'))==2:
        date = date.split('-')
    elif len(date.split('.'))==2:
        date = date.split('.')
    if int(date[0]) == 4:
        out = ' tháng tư năm '+ int_to_vn(int(date[1]))
    else:
        out = ' tháng ' + int_to_vn(int(date[0])) + ' năm ' + int_to_vn(int(date[1]))
    return out+end+' '
def _ddmm(m):
    text = m.group(0).strip()
    end = ''
    if text[-1] in ',.?!':
        end = text[-1]
        text = text[:-1]
    out = ''
    if len(text.split('/')) == 2:
        date = text.split('/')
    elif len(text.split('-')) == 2:
        date = text.split('-')
    elif len(text.split('.')) == 2:
        date = text.split('.')
    out += ' ' + int_to_vn(int(date[0])) + ' tháng ' + (int_to_vn(int(date[1])) if int(date[1]) != 4 else "tư")
    return out+end+' '
def _ddmm1(m):
    out = m.group(0).strip()
    out = re.sub(re.compile(r'([1-9]|[0-3][0-9])(/|-|\.)((0[1-9]|1[0-2])|[1-9])'), _ddmm, out)
    return out
def _days(m):
    out = m.group(0)
    out = out.replace('-', ' đến ')
    out = re.sub(re.compile(r'([0-3][0-9]|[1-9])(/|-|\.)((0[1-9]|1[0-2])|[1-9])(/|-|\.)[1-2][0-9][0-9][0-9]'), _ddmmyy, out)
    out = re.sub(re.compile(r'([1-9]|[0-3][0-9])(/|-|\.)((0[1-9]|1[0-2])|[1-9])'), _ddmm, out)
    return out+ ' '
def _phay(m):
    return m.group(0)+','
def _3G(m):
    out = m.group(0)
    out = re.sub(re.compile(r'[a-z0-9\+]+'), _phay, out)
    return out.replace('3g', 'ba gờ').replace('4g', 'bốn gờ').replace('5g', 'năm gờ')
def _duration(m):
    text = m.group(0).split('-')
    return text[0]+' đến ngày '+text[1]
def _vi(m):
    out = m.group(0)
    v = {'/':'trên', 't':'tê', 'g':'giê', 'q':'quy', 'đ':'đê', 'c':'xê', 'p':'pê', 'k':'ca', 'h':'hắc', 'v':'vê', 'b':'bê',}
    result = ' '
    for c in out:
        if c in v:
            result += v[c]+' '
    return result
def _TW(m):
    out = m.group(0)
    out = out.replace('-', '')
    out = re.sub(re.compile('(/[tgqđcpkhvb][tgqđcpkhvb]+)'), _vi, out)
    return out.replace('/', ' trên ')
def _am(m):
    out = m.group(0)
    out = out.replace('-', 'âm ')
    return out
def _name(m):
    out = m.group(0)
    out = out.replace('m4u', 'em pho du').replace('365', '3 6 5')
    return out
def _am_pm(m):
    out = m.group(0)
    out = out.replace('am', 'sáng')
    if out[-2:] == "pm":
        h = int(out[:2].strip())
        if (h > 12 and h < 18) or (h >= 1 and h < 6):
            out = out.replace('pm', 'chiều')
        elif (h >= 18 and h < 22) or (h >= 6 and h < 10):
            out = out.replace('pm', 'tối')
        elif (h >= 22 and h <= 24) or (h >= 10 and h <= 12):
            out = out.replace('pm', 'khuya')
    return out
def _noun(m):
    out = m.group(0)
    out = out.replace('\'s', ' is ')
    out = out.replace('\'re', ' are ')
    return out
def _self(m):
    return m.group(0).replace('\'', '')
def _upper(m):
    out = m.group(0).strip()
    end = ''
    if out[-1] in [',','.','?','!', ';', ':']:
        end = out[-1]
        out = out[:-1].strip()
    if out in acronym:
        return ' '+acronym[out]+end+' '
    out = ' '.join(list(out))
    return ' '+out+end+' '
def _space(m):
    out = m.group(0)
    out = out.replace('-', ' , ')
    return out
def _nay(m):
    out = m.group(0)
    out = out.replace('(', '').replace(')','')
    return out
def _AI(m):
    out = m.group(0)
    out = out.replace('AI', 'ây ai')
    return out
def _hyphen(m):
    out = m.group(0)
    return out.replace(' ', '')
def _fourth(m):
    out = m.group(0)
    return out.replace('4', ' tư ')
def _part(m):
    out = m.group(0)
    return out.replace('p', 'phần ')
_alphabet =      'aăâbcdđeêghiklmnoôơpqrstuưvxyfjwz' \
                   'áắấéếíóốớúứý' \
                   'àằầèềìòồờùừỳ' \
                   'ảẳẩẻểỉỏổởủửỷ' \
                   'ãẵẫẽễĩõỗỡũữỹ' \
                   'ạặậẹệịọộợụựỵ '


def processSent(sent):
    '''
    Thể thao 24/7 Hôm nay là 24/7 Địa chỉ là 24/7 24/7/2017 7/2017
    24,7 24.700.000 24$ 24% 24x7 24h 23h7 24m 24g 24kg 24ha 24m2 24m3 U19 ()
    :param input:
    :return:
    '''
    # Acronym & vocab & number
    _characters = '!,.?'
    input = re.sub(re.compile(r"(^|\ )(AI)(,|;|:|\?|!|\.|\ |$)"), _AI, sent)
    input = re.sub(re.compile(r'(\ )[A-Z]+[0-9]+(\ |\.|,|/|-)'), _split2, ' ' + input + ' ')
    input = ' '+ re.sub(re.compile(r"(^|\ )(AI|SE|IS|IPA|US|UK|KRQE|FBI|UAV|UAE|USS|DSCA|AM|CISM|AIoT|COC|TAC|5K|FCA|HoSE|TNHH MTV|ĐĐ.Thích)(,|\.|\?|!|;|:|\ |$)"), _upper, input+' ').lower() + ' '
    input = re.sub(re.compile(r"(bks |biển kiểm soát |biển số )[0-9][0-9][a-z][1-9]\ [0-9]+"),_license_plate,  input)
    input = re.sub(re.compile(r'từ ([0-9]|[0-3][0-9])/([0-9]|[0-1][0-9])((/([1-2][0-9][0-9][0-9]))|)((\ -\ )|(-\ )|(\ -)|(-))([0-9]|[0-3][0-9])/([0-9]|[0-1][0-9])((/([1-2][0-9][0-9][0-9]))|)'), _interval, input)
    input = re.sub(re.compile(r'(hôm nay|sáng nay|tối nay|sớm nay|chiều nay|trưa nay|ngày mai|mai)\ (\(([1-9]|[0-3][0-9])(/|-|\.)((0[1-9]|1[0-2])|[1-9])\))'), _nay, input)
    input = re.sub(re.compile(r'(nhóm|nhạc|minh vương|dương)(\ )(m4u|365|565)'), _name, input)
    input = re.sub(re.compile(r"(he|she|it|you|we|they)\'(s|re)"), _noun, input)
    input = re.sub(re.compile(r"\ [a-z]+\'s"), _self, ' '+ input)
    input = re.sub(re.compile(r"\(p[0-9]+\)"), _part, input)
    input = re.sub(re.compile(r'(quận |thứ |hạng )4(\.|,|\?|!|\ |$)'), _fourth, input)
    input = input.replace('i\'m', 'i m')
    input = input.replace('i\'d', 'i d')
    input = input.replace('p!nk', 'pink')
    input = input.replace('*', '')
    input = input.replace(';', '.')
    input = input.replace('?.', '?') #fix bug split sentence
    input = input.replace('!.', '!') #fix bug split sentence
    input = input.replace('“', '')
    input = input.replace('”', '')
    input = input.replace('\"', '')
    input = input.replace('\'s', '')
    input = input.replace('\'', '')
    input = input.replace(')', ',')
    input = input.replace('(', ', ')
    input = input.replace('″', '')
    input = input.replace('’', '')
    input = input.replace('‘', '')
    input = input.replace('#', '')
    input = input.replace('[', '')
    input = input.replace(']', '')
    #input = input.replace(',...', ', vân vân. ')
    input = input.replace('...', '. ')
    #input = input.replace(',…', ', vân vân. ')
    input = input.replace('…', '. ')
    input = input.replace('=', ' bằng ')
    input = re.sub(re.compile(r'(,)(\ |,)+'), ', ', input)
    input = re.sub(re.compile(r'[0-9][0-9\ ]*(mỗi|từng|chục|trăm|tỷ|nghìn|ngàn|triệu|đồng|đơn vị|đơn vị là|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mươi|lăm|mốt|\ )*(mm|khz|m3/s|đ/kg|£|mah|€|đồng/kg|đồng/km|kg|ha|mmhg|vnđ|ndt|vnd|µm|ft|m/s|km/h|gb|hz|mhz|m²|độ c|\$|%|km²|km|m³|ms|kwh|mw|mg|cm|°f|°c|m2|km2|m3|ml|l|kw|mm|nm)[/\ ,.?!-]'), _measure, input+' ')
    input = re.sub(re.compile(r'(quyết định|nghị định|thông tư|văn bản|nghị định|số)(\ )([0-9][0-9/]*[tgqđcpkhvb][tgqđcpkhvb\-]*)'), _TW, input)
    input = re.sub(re.compile(r'(^|\ )\-[0-9]+'), _am, input)
    input = re.sub(re.compile(r'(\ )[a-zđ]+[0-9]+(\ |\.|,|/|-)'), _split, ' ' + input + ' ')
    input = re.sub(re.compile(r'(www.|http://|https://|)[a-z0-9\.\-]+@(gmail|yahoo|outlook|olli-ai)(\.com|\.vn|\.edu)+'), _mail, input)
    input = re.sub(re.compile(r'(www.|http://|https://|)[a-z0-9\.\-]+(\.com|\.vn|\.edu|\.org|\.net)+'), _link, input)
    input = re.sub(re.compile(r"(thế kỉ|thứ|giải|hạng|bsck|quý|khóa|khoá|khoa|tầng|chương|mục|phần|nhóm|đại hội|tk|đệ|thế chiến|cấp|kỳ|kì|kỷ)\ [ivx]+[\ .,/-]"),_roman_numerals, input)
    input = re.sub(re.compile(r'[1-9][0-9\ ]*x((\ |)*[1-9][0-9]*)[\ .,-/]'), _size, input+' ')
    #print('in ', input)
    input = re.sub(re.compile(r"(lúc|từ|khoảng|đến|tới|vào|hồi)\ ([1-9]|[0-2][0-9])((\ :\ )|(\ :)|(:\ )|:)(([0-6][0-9]|[1-9])|)((((\ :\ )|(\ :)|(:\ )|:)([0-6][0-9]|[1-9])){1,2})(\ |\.|,|-||–| –|)"), _hour, input + ' ')
    #print('out ', input)
    input = re.sub(re.compile(r"([0-9]|[0-2][0-9])((\ h\ )|(h\ )|(\ h)|(\ h:\ )|(\ h:)|(h:\ )|(h:)|h)((([0-9]|[0-6][0-9])(p|))|)(:|\.|,|-| -|–| –|\ )"), _hour_minute, input.strip()+' ')
    input = re.sub(re.compile(r"([0-9]|[0-2][0-9])((\ :\ )|(\ :)|(:\ )|:)([0-9]|[0-6][0-9])(p|)(:|\.|,|-| -|–| –|\ )"), _hour_minute, input.strip()+' ')
    #print(input)
    input = re.sub(re.compile(r"(lúc|từ|đến|tới|vào|hồi)\ ([0-9]|[0-2][0-9])((\ g\ )|(\ g)|(\ g)|g)(\.|,|-| -|–| –|\ )"), _hour_minute1, input+' ')
    #print(input)
    input = re.sub(re.compile(r"([0-9]|[0-2][0-9])((\ g\ )|(\ g)|(\ g)|g)((([0-9]|[0-6][0-9])(p|)))(\.|,|-| -|–| –|\ )"), _hour_minute, input+' ')
    print(input)
    input = re.sub(re.compile(r"([0-9]|[0-2][0-9])((\ g:\ )|(\ g:)|(\ g:)|g:)((([0-9]|[0-6][0-9])(p|))|)(:|\.|,|-| -|–| –|\ )"), _hour_minute, input+' ')
    #print(input)
    input = re.sub(re.compile(r"(khoảng\ )([0-9]|[0-2][0-9])((\ g\ )|(\ g)|(g\ )|(\ g:\ )|(\ g:)|(g:\ )|(g:)|g)((([0-9]|[0-6][0-9])(p|))|)(\ )(một lần|mỗi ngày|hàng ngày|cùng ngày|ngày|trưa|tối|sáng|rạng sáng|buổi)"), _hour_minute1, input+' ')
    input = re.sub(re.compile(r'[0-9][0-9\ ]*k[/\ .,-]'), _thounsand, input)
    input = re.sub(re.compile(r'\ ((p(\ )[0-9]{1,2})|(p\.(\ )*([0-9]{1,2}|[a-zđ]{1,})))'), _ward, ' ' + input)
    input = re.sub(re.compile(r'\ ((q(\ )[0-9]{1,2})|(q\.(\ )*([0-9]{1,2}|[a-zđ]{1,})))'), _district, ' ' + input)
    input = re.sub(re.compile(r'\ (tp\.|t\.p\ |tp\. |tt\.|tx\.|tt\. |tx\. )[đâơôa-z]'), _city, ' '+ input)
    input = re.sub(re.compile(r'\ [1-9][0-9][0-9]\.([0-9][0-9][0-9]\.)*[0-9][0-9][0-9]\ '), _bignum, ' '+input)
    input = re.sub(re.compile(r'[0-9][0-9\ ]+đ[/\ .,-]'), _money, input+' ')
    input = re.sub(re.compile(r'\ ([a-z0-9\+]+|dung lượng|tài khoản|gói cước|sim|lưu lượng|đăng ký|nạp tiền|gia hạn|mạng|dịch vụ|sử dụng|truy cập|kết nối)\ (là |)(3|4|5)g[\ \.,-?!]'), _3G, ' ' + input + ' ')
    input = re.sub(re.compile(r'[0-9][0-9\ ]*g[/\ .,-]'), _weight, input + ' ')
    input = re.sub(re.compile(r'[0-9][0-9\ ]*l[/\ .,-]'), _volume, input + ' ')
    input = re.sub(re.compile(r'[0-9][0-9\ ]*m[\ .,-]'), _m, input + ' ')
    input = re.sub(re.compile(r'[0-9][0-9\ ]*v[\ .,-]'), _v, input + ' ')
    input = re.sub(re.compile(r'((no|vol)\.)(\ |)[0-9]{1,2}'), _no, input)
    input = re.sub(re.compile(r'(ngày|đêm|tối|sáng|trưa|chiều|từ)\ (([1-9]|[0-3][0-9]))((\ |)(-|và|đến|tới)(\ |))(((([0-3][0-9]|[1-9])(/|\.)((0[1-9]|1[0-2])|[1-9])(/|\.)[1-2][0-9][0-9][0-9]))|(([1-9]|[0-3][0-9])(/|\.|-)(0[1-9]|1[0-2]|[1-9])))'),_days, input + ' ')
    input = re.sub(re.compile(r'từ ([1-9]|[0-3][0-9])(/|\.)((0[1-9]|1[0-2])|[1-9])(\ |,|;|\.)'), _ddmm1, input)
    # 8/9/2018, 8-9-2018, 8.9.2018
    input = re.sub(re.compile(r'([0-3][0-9]|[1-9])/((0[1-9]|1[0-2])|[1-9])/[1-2][0-9][0-9][0-9]'), _ddmmyy, input)
    input = re.sub(re.compile(r'([0-3][0-9]|[1-9])\.((0[1-9]|1[0-2])|[1-9])\.[1-2][0-9][0-9][0-9]'), _ddmmyy, input)
    input = re.sub(re.compile(r'([0-3][0-9]|[1-9])-((0[1-9]|1[0-2])|[1-9])-[1-2][0-9][0-9][0-9]'), _ddmmyy, input)
    # # 9.2018, 9-2018, 9/2018, ngày 7-9, 14-9, 21-8
    input = re.sub(re.compile(r'tháng\ ((0[1-9]|1[0-2])|[1-9])(/|\.|-)[1-2][0-9][0-9][0-9]'), _mmyy, input + ' ')
    input = re.sub(re.compile(r'\ ((0[1-9]|1[0-2])|[1-9])(/|\.|-)[1-2][0-9][0-9][0-9](\ |\.|,)'), _mmyy, input + ' ')
    input = re.sub(re.compile(r'(([0-9]|[0-2][0-9])(\ giờ)((\ ([0-9]|[0-6][0-9]) phút)|\ ([0-9]|[0-6][0-9])|)((\ ([0-9]|[0-6][0-9]) giây)|))(\ )(am|pm)'), _am_pm, input)
    input = input.replace(':', ', ')
    out =''
    words = ViTokenizer.tokenize(input)
    words = words.replace(' & ', '&')
    words = re.sub(re.compile(r'[0-9](\ )*\-(\ )*[0-9]'), _hyphen, words)
    words = words.replace(' .', '.')
    words = words.replace('thứ hai', 'thứ_hai')
    words = words.replace('thứ ba', 'thứ_ba')
    words = words.replace('thứ tư', 'thứ_tư')
    words = words.replace('thứ năm', 'thứ_năm')
    words = words.replace('thứ sáu', 'thứ_sáu')
    words = words.replace('thứ bảy', 'thứ_bảy')
    words = words.replace('thứ 2 ', 'thứ_hai ')
    words = words.replace('thứ 3 ', 'thứ_ba ')
    words = words.replace('thứ 4 ', 'thứ_tư ')
    words = words.replace('thứ 5 ', 'thứ_năm ')
    words = words.replace('thứ 6 ', 'thứ_sáu ')
    words = words.replace('thứ 7 ', 'thứ_bảy ')
    words = words.replace('chủ nhật', 'chủ_nhật')
    words = words.replace('số nhà', 'số_nhà')
    words = words.replace('giổ tổ', 'giổ_tổ')

    dates = ['thứ_hai','thứ_ba','thứ_tư','thứ_năm','thứ_sáu','thứ_bảy','chủ_nhật', 'thứ_2', 'thứ_3', 'thứ_4', 'thứ_5', 'thứ_6', 'giổ_tổ', 'mùng',
             'ngày', 'tháng', 'sáng', 'trưa', 'chiều','tối', 'qua', 'mai', 'đêm', 'vào', 'khủng_bố', 'sự_kiện',
             'khuya', 'hôm', 'quốc_khánh', 'lễ', 'thiếu_nhi', 'việt_nam', 'nay', 'đến', 'phiên', 'hôm_nay', 'ngày_mai']
    spec_day = ['2/9','8/3', '3/2', '20/11', '30/4', '1/5', '10/3', '27/7', '22/12']
    address = ['địa_chỉ', 'hẻm', 'số_nhà', 'ngõ', 'đường']
    for id, word in enumerate(words.split(' ')):
        if word in spec_day or (id > 0 and (words.split(' ')[id - 1] in dates) and word[0].isdigit()):
            end = ''
            if word[-1] == '.' or word[-1] ==',':
                end = word[-1]
                word = word[:-1]
            word = re.sub(re.compile(r'([1-9]|[0-3][0-9])-([1-9]|[0-3][0-9])(/|\.)((0[1-9]|1[0-2])|[1-9])'), _duration, word)
            word = re.sub(re.compile(r'([1-9]|[0-3][0-9])(/|-|\.)((0[1-9]|1[0-2])|[1-9])'), _ddmm, word)
            out += word.strip() + end + ' '
        elif len(word.split('/')) > 1:
            for id1, w in enumerate(word.split('/')):
                if w.isdigit():
                    if id1 != len(word.split('/')) - 1 and words.split(' ')[id - 1] in address:
                        out += int_to_vn(int(w)) + ' siệt '
                    elif id1 != len(word.split('/')) - 1 and words.split(' ')[id - 1] not in address:
                        out += int_to_vn(int(w)) + ' trên '
                    else:
                        out += int_to_vn(int(w)) + ' '
                else:
                    if id1 != len(word.split('/')) - 1:
                        out += w + ' trên '
                    else:
                        out += w + ' '
        elif len(word) >2 and len(word.split('-')) == 2 and word.split('-')[0][0].isdigit() and word.split('-')[1][0].isdigit():
            if id - 1 >= 0 and words.split(' ')[id - 1] in ['từ', 'khoảng', 'tầm' , 'ngày']:
                word = word.replace('-', ' đến ')
            out += word+' '
        else:
            word = re.sub(re.compile(r'[0-9][0-9\.]*\,[0-9]+'), _float, ' ' + word + ' ')
            word = word.strip()
            end = ''
            if len(word) > 0 and word[-1] in _characters and word not in ['mr.']:
                end = word[-1]
                word = word[:-1]
            if word in acronym:
                word = acronym[word]
            out += word + end + ' '
    tokens = ViTokenizer.tokenize(out.strip())
    tokens = tokens.replace('_', ' ')
    tokens = tokens.replace('/', ' ')
    tokens = tokens.replace('\\', ' ')
    tokens = tokens.replace(', ,', ' , ')
    tokens = tokens.replace('&', ' và ')
    tokens = tokens.replace('+', ' cộng ')
    tokens = tokens.replace('giờ 00', ' giờ ')
    tokens = re.sub(re.compile(r'[0-9]+-[0-9]+'), _space, tokens)
    tokens = tokens.replace(' - ', ' , ')
    tokens = tokens.replace('-', ' ')
    tokens = tokens.replace(' –', ' ')
    tokens = tokens.replace('– ', ' ')
    tokens = re.sub(re.compile(r'\ [0-9][0-9\.]*\,[0-9]+'), _float, ' ' + tokens + ' ')

    tokens = re.sub(re.compile(r'[0-9]+(\.[0-9]{3})+[\ \.,?!]'), _dot, tokens+' ')
    tokens = re.sub(re.compile(r'[0-9]+(\.)0[0-9]*'), _dot3, tokens)
    tokens = re.sub(re.compile(r'[0-9]+((\.)[0-9]+)+'), _dot4, tokens)

    tokens = re.sub(re.compile(r"\ (tổng đài|liên hệ|số điện thoại)(\ )*(1800|1900|0)[0-9\.\ ]+"), _phone2, tokens)
    tokens = re.sub(re.compile(r"(\ )*(ngày|tháng|số|thứ)(\ )*([0-9]+)"), _num, tokens)
    tokens = re.sub(re.compile(r"\ (0|\+8)[0-9\.]{8,9}"), _phone, tokens)
    tokens = re.sub(re.compile(r"\ [0-9]+[a-zđ\-]+(\ |\.|,)"), _split4, tokens)
    tokens = re.sub(re.compile(r'[a-zđ](\.[a-zđ])+'), _dot2, tokens)
    result = ''
    for token in tokens.split(' '):
        if token.isdigit():
            result += int_to_vn(int(token))+ ' '
        elif token in _characters:
            result = result[:-1] + token + ' '
        elif token in acronym:
            result += acronym[token] + ' '
        elif token != '':
            result += token + ' '
    result = re.sub(re.compile(r'\ ([bcdđghklmnpqrstvxfjwz0-9]{2,})+[\ \.,?!]'), _split3, ' ' + result + ' ')
    result = re.sub(re.compile(r'\ ([aeoyiu]+[0-9]+)([bcdđghklmnpqrstvxfjwz]+|)[\ \.,?!]'), _split3, ' ' + result + ' ')
    result = result.replace('_',' ').strip()
    result = ' '.join([x for x in result.split(' ') if x != ''])

    if len(result) >= 1 and result[-1] not in _characters:
        result = result + '.'

    return result.strip()
#print(processSent("biện pháp 5K trốn tìm Đen, MTV, cty TNHH MTV olli"))

stop_word = [
    'và', 'sau_khi', 'khi', 'bởi', 'vì_sao', 'điều_này', 'cho_rằng',
    'rằng', 'nếu', 'vì', 'lúc_này', 'khi_đó', 'nên', 'cũng_như', 'mà', 'tại_sao',
    'lúc', 'vậy', 'tại_sao', 'một_cách', 'đến_nỗi', 'bởi_vì',
    'do_đó', 'do', 'sau_đó', 'đó_là', 'thế_nhưng', 'tại', 'thì', 'hoặc', 'với',
    'tất_nhiên', 'đương_nhiên', 'thay_vì', 'vì_vậy', 'giả_sử', 'giá_như', 'nhưng',
    'may_mà', 'thế_mà', 'tuy', 'rằng', 'mặc_dù', 'hễ', 'hèn_chi', 'so_với',
    'huống_chi', 'huống_hồ', 'vả_lại', 'họa_chăng', 'kẻo', 'kẻo_mà', 'kẻo_nữa',
    'hèn_chi', 'hèn_gì', 'thảo_nào', 'để', 'giả_sử', 'ví_như', 'dường_như', 'dẫu', 'tuy',
    'ví_như', 'tuy_rằng', 'thế_mà', 'mà', 'vậy_mà', 'thế_mà', 'dẫu', 'thì', 'huống_hồ', 'biết_đâu', 'quả nhiên',
    'bởi_vậy', 'thành_thử', 'còn_như', 'kẻo_lại', 'vậy_mà',
    'thế_thì', 'huống_là', 'hay_là', 'miễn_là', 'dù', 'như_vậy', 'đến_khi', 'cho_đến', 'đến_nỗi',
    'trong_khi', 'trong_lúc', 'thảo_nào', 'trong', 'dẫn_đến', 'bao_gồm', 'sau_đó',

]
def pretokenize(doc):
    doc = doc.replace('sau khi', 'sau_khi')
    doc = doc.replace('tại sao', 'tại_sao')
    doc = doc.replace('so với', 'so_với')
    doc = doc.replace('bởi vậy', 'bởi_vậy')
    doc = doc.replace('thành thử', 'thành_thử')
    doc = doc.replace('còn như', 'còn_như')
    doc = doc.replace('kẻo lại', 'kẻo_lại')
    doc = doc.replace('vậy mà', 'vậy_mà')
    doc = doc.replace('huống là', 'huống_là')
    doc = doc.replace('hay là', 'hay_là')
    doc = doc.replace('miễn là', 'miễn_là')
    doc = doc.replace('cho đến', 'cho_đến')
    doc = doc.replace('đến khi', 'đến_khi')
    doc = doc.replace('đến nổi', 'đến_nổi')
    doc = doc.replace('trong khi', 'trong_khi')
    doc = doc.replace('trong lúc', 'trong_lúc')
    doc = doc.replace('dẫn đến', 'dẫn_đến')
    doc = doc.replace('cho rằng', 'cho_rằng')
    doc = doc.replace('một cách', 'một_cách')
    doc = doc.replace('điều này', 'điều_này')
    doc = doc.replace('cũng như', 'cũng_như')
    doc = doc.replace('sau đó', 'sau_đó')


    return doc

def split(doc):

    max_len=35
    sent_list = nltk.sent_tokenize(doc)
    out_list = []

    for sent in sent_list:
        if len(sent.split(' ')) <= max_len:
            out_list.append(sent)
        else:
            clause = re.split(", |\?|!|,|\.", sent)
            for seq in clause:
                word_list = seq.split(' ')
                if len(word_list)<=max_len:
                    out_list.append(seq)
                else:
                    chk = ViTokenizer.tokenize(seq)
                    chk = pretokenize(chk).split()
                    start = 0
                    for index, token in enumerate(chk):
                        if token in stop_word and index != len(chk)-1:
                            out_list.append(' '.join(chk[start:index]).replace('_', ' '))
                            start = index
                        elif index == len(chk)-1:
                            out_list.append(' '.join(chk[start:index+1]).replace('_', ' '))
    return out_list

def split1(doc):
    max_len=35
    sent_list = nltk.sent_tokenize(doc)
    out_list = []

    for sent in sent_list:
        sent = sent.strip()
        if len(sent.split(' ')) <= max_len:
            out_list.append(sent)
        else:
            p_list = []
            clause = re.split(", |\?|!|,", sent)
            for seq in clause:
                word_list = seq.strip().split(' ')
                if len(word_list)<=max_len:
                    p_list.append(seq)
                else:
                    chk = ViTokenizer.tokenize(seq)
                    chk = pretokenize(chk).split()
                    start = 0
                    for index, token in enumerate(chk):
                        if token in stop_word and index != len(chk)-1:
                            p_list.append(' '.join(chk[start:index]).replace('_', ' '))
                            start = index
                        elif index == len(chk)-1:
                            p_list.append(' '.join(chk[start:index+1]).replace('_', ' '))
            p_list = [x.strip() for x in p_list if x != '']
            part = partition(p_list)
            text_split = sent.split(' ')
            text_split = [x.strip() for x in text_split if x != '']
            id = 0
            for i in part:
               out_list.append(' '.join(text_split[id:(id+i)]))
               id +=i
    return out_list
