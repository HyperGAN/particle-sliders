"""Same deployed generation loop, with independently budgeted host components."""
from ..reward_search.renderer import SearchRenderer


class FeedForwardRenderer(SearchRenderer):
    def components(self,family,extra=None):
        components=[dict(c) for c in self.manifest['style_components'][family['family']]]
        if extra:components.append(dict(extra))
        energy=dict(language_model=0.,transformer=0.)
        for c in components:energy[c['kind']]+=abs(c['multiplier']*c['alpha']/c['rank'])
        for kind,value in energy.items():
            if value>self.manifest['host_energy_by_kind'][kind]+1e-8:raise ValueError('Excess '+kind+' energy')
        return components
