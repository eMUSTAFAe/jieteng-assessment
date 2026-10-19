import time, json
def adapter(text): return text.upper()
def run(cases):
    rows=[]
    for case in cases:
        start=time.perf_counter(); output=adapter(case['input']); rows.append({'output':output,'correct':output==case['expected'],'latency_ms':(time.perf_counter()-start)*1000})
    return rows
if __name__ == '__main__': print(json.dumps(run([{'input':'hello','expected':'HELLO'}]),indent=2))
