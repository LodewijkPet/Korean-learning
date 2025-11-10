import asyncio, edge_tts
async def m():
    vm = await edge_tts.VoicesManager.create()
    for v in vm.voices:
        if v['Locale'] == 'ko-KR':
            print(v['ShortName'])
asyncio.run(m())
