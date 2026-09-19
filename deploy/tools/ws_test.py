import asyncio, json, struct, sys, time, glob, os
import numpy as np, soundfile as sf, torch, torchaudio, websockets
async def turn(ws, tid, pcm=None, text=None, cancel_after=None):
    t=time.perf_counter()
    if pcm is not None: await ws.send(struct.pack("<I",tid)+pcm.tobytes())
    else: await ws.send(json.dumps({"type":"text_turn","id":tid,"text":text}))
    first=None; nbytes=0; txt=""; done=None; cancelled=False
    while True:
        m=await asyncio.wait_for(ws.recv(),60)
        if isinstance(m,bytes):
            nbytes+=len(m)-4
            if first is None: first=(time.perf_counter()-t)*1000
            if cancel_after and not cancelled and nbytes/2/22050>=cancel_after:
                await ws.send(json.dumps({"type":"cancel","id":tid,"keep":True})); cancelled=True
        else:
            j=json.loads(m)
            if j["type"]=="text": txt=j["text"]
            if j["type"]=="done": done=j; break
            if j["type"]=="error": print("ERR",j); break
    print(f"turn {tid}: first-audio-byte {first and round(first)}ms  audio {nbytes/2/22050:.1f}s  total {(time.perf_counter()-t)*1000:.0f}ms  interrupted={(done or {}).get('interrupted')}")
    print("   text:",(done or {}).get("text",txt)[:200]); return done
async def main():
    async with websockets.connect("ws://127.0.0.1:8890/ws",max_size=None) as ws:
        print(await ws.recv())
        f=sorted(glob.glob("/workspace/captures/glm4voice/reply-*.wav"),key=os.path.getmtime)[-1]
        d,sr=sf.read(f,dtype="float32"); d=torchaudio.functional.resample(torch.from_numpy(d),sr,16000).numpy()
        pcm=(np.concatenate([d,np.zeros(4000,np.float32)])*32767).astype("<i2")
        await turn(ws,1,text="Give me a fun fact about space.")
        await turn(ws,2,text="Hello, how are you today?")
        await turn(ws,3,pcm=pcm)
        await turn(ws,4,text="Tell me a long story about a dragon.",cancel_after=1.5)
        await turn(ws,5,text="What did I just ask you about?")
asyncio.run(main())
