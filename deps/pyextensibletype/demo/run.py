import provider
import consumer

class ProviderSubclassOnHeap(provider.Provider):
    pass

print consumer.sum_baseline(provider.Provider(), 4)
print consumer.sum_baseline(object(), 4)
print consumer.sum_baseline(ProviderSubclassOnHeap(), 4)

