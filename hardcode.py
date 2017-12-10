#!/usr/bin/python3
# -*- coding: utf-8 -*-

def check_hardcode(req):
    text = req['messages'][-1][1]
    if any(x.lower() in text.lower() for x in [u"привет",u"хай",u"добрый",u"здравствуйте"]):
        return u"добрый день, "+req["name"]
    return ""


def check_hard_calc(req):
    text = req['messages'][-1][1]
    if any(x.lower() in text.lower() for x in [u"каско",u"осаго",u"авто",u"машин",u"hundai",u"kia",u"иномарк"]):
        if any(x.lower() in text.lower() for x in [u"рассчет",u"стоит",u"почем",u"скольк",u"купит",u"взят",u"дай",u"посчит",u"офор"]):
            return True
    return False
