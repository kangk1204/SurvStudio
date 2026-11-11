# Third-party notices

SurvStudio's original software is licensed under MIT (see LICENSE). This does not relicense third-party components or datasets.

## Bundled assets

| Component | Bundled path | License and full text |
| --- | --- | --- |
| Sora | src/survival_toolkit/static/fonts/Sora-*.woff2 | SIL Open Font License 1.1, licenses/Sora-OFL.txt |
| Fraunces | src/survival_toolkit/static/fonts/Fraunces-*.woff2 | SIL Open Font License 1.1, licenses/Fraunces-OFL.txt |
| Plotly.js 3.4.0 | src/survival_toolkit/static/vendor/plotly-3.4.0.min.js | MIT, licenses/Plotly-LICENSE.txt; embedded components retain their licenses |

Bundled Sora font metadata: Copyright 2019 The Sora Project Authors (https://github.com/sora-xor/sora-font).
Bundled Fraunces font metadata: Copyright 2020 The Fraunces Project Authors (github.com/undercasetype/Fraunces).
The upstream Fraunces OFL file also retains its original 2018 copyright notice. Font files and embedded notices are unmodified.
The Plotly bundle header retains Copyright 2012-2026, Plotly, Inc.; the upstream full MIT file retains Copyright 2016-2024 Plotly Technologies Inc.

Plotly's preserved bundle notices identify Native Promise Only (Kyle Simpson), polybooljs (Sean Connelly), buffer, safe-buffer, is-buffer and ieee754 (Feross Aboukhadijeh), object-assign (Sindre Sorhus), pad-left and repeat-string (Jon Schlinkert), and MapLibre GL JS. Full license texts are included in licenses/. MapLibre's file also contains its Mapbox, glfx.js and d3-color notices. The assertion utility's retained buffer attribution is covered by the buffer license text. MIT notices are not a claim that BSD or OFL components are MIT-licensed.

Sources: Google Fonts ofl/sora/OFL.txt and ofl/fraunces/OFL.txt; Plotly.js v3.4.0 LICENSE; MapLibre GL JS v4.7.1 LICENSE.txt; component upstream license files. Native Promise Only's bundled copyright notice and published MIT designation accompany the full standard MIT permission and disclaimer text. Other original copyright notices remain in the bundle and fonts.

## Separately installed dependencies

Python dependencies are not bundled in this source archive. They retain their own licenses. In particular, the optional ML/all/validation dependency scikit-survival uses GPL-3.0 (https://github.com/sebp/scikit-survival/blob/main/COPYING). Installing this optional dependency does not make it MIT-licensed. Redistribution of a combined installation must account for the applicable dependency licenses and source-availability requirements; the SurvStudio MIT notice alone is not a license for that combined distribution.

## Data

The MIT license for SurvStudio code does not grant rights to third-party example datasets. See [Example data](examples/README.md) for sources and check the original provider's terms before redistributing those data.
