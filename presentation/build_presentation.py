"""Build the 2-minute presentation deck and its appendix PDF from committed results.

Every number on the slides is read from the repository's generated result files:
results/benchmark/summary.json (12 systems, extended rows, paired comparison),
results/data_integrity.json (cohorts, cleaning, overlap) and the scored test
utterances of the HParl+CSS10 Whisper fine-tune (real error example).

    python presentation/build_presentation.py

Writes presentation/greek_parliamentary_asr.pptx (with speaker notes) and
presentation/appendix.pdf (full table, metrics, split report, reproduction).
Fonts: Bricolage Grotesque, Figtree and DM Mono (SIL OFL, presentation/fonts/);
Greek text uses Arial because these fonts have no Greek glyphs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.dml import MSO_LINE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'presentation'
EXAMPLE_SYSTEM = 'whisper_mixed'
EXAMPLE_ID = 'hparl:alekshs_tsipras-olomeleia_1_00365000_00368000_5_1'
EXAMPLE_GLOSS = '"who benefits from this conflict"'
REPO_URL = 'github.com/kaanksn1/greek-parliamentary-asr'

# Design units: a 680 x 383 canvas mapped onto a 13.333 x 7.5 inch slide.
SLIDE_W_IN, CANVAS_W = 13.333, 680
EMU_PER_UNIT = SLIDE_W_IN * 914400 / CANVAS_W
PT_PER_UNIT = SLIDE_W_IN * 72 / CANVAS_W

HEAD, BODY, MONO, GREEK = 'Bricolage Grotesque', 'Figtree', 'DM Mono', 'Arial'
BG, CARD, CARD_LINE, NAVY, INK, MUTED, CUE = 'F6F1E7', 'FFFCF5', 'E2D8C4', '14213D', '2B2B2B', '7D7566', 'B8AE9A'
BLUE_DARK, RED, GRID = '2E4A7D', 'B03A2E', 'D3C9B5'
FAMILY = {'classical': ('Classical', '2A6FDB'), 'scratch': ('Trained from scratch', 'E8622C'),
          'ssl': ('SSL encoder + CTC', '7C4DDB'), 'pretrained': ('Pretrained ASR', '1F8A5B'),
          'contribution': ('Our contribution', 'C2185B')}
SYSTEMS = {
    'gmm_hmm': ('MFCC + GMM-HMM', 'classical'), 'plain_ctc': ('Plain CTC', 'scratch'),
    'attention': ('Attention enc-dec', 'scratch'), 'transducer': ('RNN-T', 'scratch'),
    'wav2vec': ('wav2vec2 XLSR Greek', 'ssl'), 'hubert_ctc': ('mHuBERT-147 + CTC (ours)', 'ssl'),
    'whisper_pretrained': ('Whisper-small', 'pretrained'), 'canary': ('Canary-1b-v2', 'pretrained'),
    'parakeet': ('Parakeet-tdt-0.6b-v3', 'pretrained'), 'qwen': ('Qwen3-ASR-0.6B', 'pretrained'),
    'whisper_hparl': ('Whisper FT HParl (ours)', 'contribution'),
    'whisper_mixed': ('Whisper FT +CSS10 (ours)', 'contribution'),
}


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def load_data():
    summary, integrity = read('results/benchmark/summary.json'), read('results/data_integrity.json')
    if summary['status'] != 'complete':
        raise ValueError('benchmark summary is not complete')
    rows = {row['system']: row for row in summary['rows']}
    extended = {row['original_system']: row for row in summary.get('extended_rows', []) if row['status'] == 'complete'}
    example = None
    with (ROOT / 'results/benchmark' / EXAMPLE_SYSTEM / 'utterances.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            if item['id'] == EXAMPLE_ID:
                example = item
                break
    if example is None:
        raise ValueError('example utterance not found')
    return summary, rows, extended, integrity, example


# --------------------------------------------------------------------------- drawing helpers
def u(value):
    return Emu(int(round(value * EMU_PER_UNIT)))


def rgb(hex_value):
    return RGBColor.from_string(hex_value)


class Canvas:
    def __init__(self, slide):
        self.slide = slide
        self.shapes = slide.shapes
        background = slide.background.fill
        background.solid()
        background.fore_color.rgb = rgb(BG)

    def rect(self, x, y, w, h, fill=None, line=None, line_width=1.0, radius=0.0, dash=False, name=None):
        kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
        shape = self.shapes.add_shape(kind, u(x), u(y), u(w), u(h))
        if radius:
            shape.adjustments[0] = min(0.5, radius / min(w, h))
        if fill:
            shape.fill.solid()
            shape.fill.fore_color.rgb = rgb(fill)
        else:
            shape.fill.background()
        if line:
            shape.line.color.rgb = rgb(line)
            shape.line.width = Pt(line_width)
            if dash:
                shape.line.dash_style = MSO_LINE.DASH
        else:
            shape.line.fill.background()
        shape.shadow.inherit = False
        if name:
            shape.name = name
        return shape

    def oval(self, cx, cy, r, fill):
        shape = self.shapes.add_shape(MSO_SHAPE.OVAL, u(cx - r), u(cy - r), u(2 * r), u(2 * r))
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)
        shape.line.fill.background()
        shape.shadow.inherit = False
        return shape

    def line(self, x1, y1, x2, y2, color, width=1.0, dash=False, arrow=False, round_cap=False):
        connector = self.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, u(x1), u(y1), u(x2), u(y2))
        connector.line.color.rgb = rgb(color)
        connector.line.width = Pt(width)
        if dash:
            connector.line.dash_style = MSO_LINE.DASH
        ln = connector.line._get_or_add_ln()
        if round_cap:
            ln.set('cap', 'rnd')
        if arrow:
            tail = etree.SubElement(ln, qn('a:tailEnd'))
            tail.set('type', 'triangle')
            tail.set('w', 'med')
            tail.set('h', 'med')
        return connector

    def bar(self, x1, x2, y, color, thickness=4.0):
        """CI bar as a thin rounded rectangle (renders identically everywhere)."""
        x1, x2 = min(x1, x2), max(x1, x2)
        if x2 - x1 < thickness:
            mid = (x1 + x2) / 2
            x1, x2 = mid - thickness / 2, mid + thickness / 2
        return self.rect(x1, y - thickness / 2, x2 - x1, thickness, fill=color, radius=thickness / 2)

    def text(self, x, y, w, h, runs, size, font=BODY, color=INK, bold=False, align='l', anchor='t', wrap=False):
        """runs: str or list of (text, overrides) tuples. y is the top of the box in canvas units."""
        box = self.shapes.add_textbox(u(x), u(y), u(w), u(h))
        frame = box.text_frame
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        frame.word_wrap = wrap
        frame.vertical_anchor = {'t': MSO_ANCHOR.TOP, 'm': MSO_ANCHOR.MIDDLE, 'b': MSO_ANCHOR.BOTTOM}[anchor]
        paragraph = frame.paragraphs[0]
        paragraph.alignment = {'l': PP_ALIGN.LEFT, 'c': PP_ALIGN.CENTER, 'r': PP_ALIGN.RIGHT}[align]
        if isinstance(runs, str):
            runs = [(runs, {})]
        for content, overrides in runs:
            run = paragraph.add_run()
            run.text = content
            style = {'font': font, 'size': size, 'color': color, 'bold': bold, **overrides}
            run.font.name = style['font']
            run.font.size = Pt(style['size'] * PT_PER_UNIT)
            run.font.bold = style['bold']
            run.font.color.rgb = rgb(style['color'])
            if style['font'] == GREEK:
                run_properties = run._r.get_or_add_rPr()
                for tag in ('a:latin', 'a:cs'):
                    element = run_properties.find(qn(tag))
                    if element is None:
                        element = etree.SubElement(run_properties, qn(tag))
                    element.set('typeface', GREEK)
        return box

    def header(self, label, title):
        self.text(28, 18, 624, 16, label, 11, font=MONO, color=MUTED)
        self.text(28, 42, 624, 36, title, 27 if len(title) <= 44 else 24.5, font=HEAD, color=NAVY, bold=True)


def hours(value):
    return f'{value:.1f} h'


def dot_plot(canvas, rows, *, left, right, top, step, axis_max, ticks, label_size, dot_r, annotate_from=440):
    """rows: list of dicts with label, family, wer, low, high, bold. Returns x mapping."""
    scale = lambda value: left + (right - left) * value / axis_max
    bottom = top + step * (len(rows) - 1) + step * 0.6
    for tick in ticks:
        canvas.line(scale(tick), top - step * 0.75, scale(tick), bottom, GRID, 0.75)
        canvas.text(scale(tick) - 20, bottom + 6, 40, 14, f'{tick:g}', 11, font=MONO, color='8A8170', align='c')
    for index, row in enumerate(rows):
        y = top + index * step
        color = FAMILY[row['family']][1]
        canvas.text(left - 214, y - label_size * 0.62, 202, label_size * 1.3, row['label'], label_size,
                    color='1F1F1F', bold=row.get('bold', False), align='r')
        canvas.bar(scale(row['low']), scale(row['high']), y, color)
        canvas.oval(scale(row['wer']), y, dot_r, color)
        value = f"{row['wer']:.2f}"
        if scale(row['high']) > annotate_from:
            canvas.text(scale(row['low']) - 62, y - 8, 54, 16, value, 11, font=MONO, color='333333', align='r')
        else:
            canvas.text(scale(row['high']) + 7, y - 8, 54, 16, value, 11, font=MONO, color='333333')
    return scale


def ci(row):
    return row['wer_ci95_low_percent'], row['wer_ci95_high_percent']


# --------------------------------------------------------------------------- slides
def slide_problem(canvas, integrity, rows):
    cohorts = integrity['cohorts']
    canvas.text(28, 18, 624, 16, 'COE025 · PROJECT 3 · GREEK PARLIAMENTARY ASR · ANONYMOUS GROUP', 11, font=MONO, color=MUTED)
    canvas.text(28, 42, 624, 36, 'Does read speech help parliamentary ASR?', 27, font=HEAD, color=NAVY, bold=True)
    canvas.rect(36, 110, 196, 84, fill=CARD, line=BLUE_DARK, line_width=1.5, radius=12, name='HParl card')
    # Parliament glyph: pediment, four columns, base.
    pediment = canvas.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, u(54), u(124), u(36), u(10))
    pediment.fill.solid(); pediment.fill.fore_color.rgb = rgb(BLUE_DARK); pediment.line.fill.background()
    for x in (57, 65, 75, 83):
        canvas.rect(x, 137, 4, 18, fill=BLUE_DARK)
    canvas.rect(54, 156, 36, 4, fill=BLUE_DARK)
    canvas.text(102, 128, 120, 22, 'HParl', 20, font=HEAD, color=NAVY, bold=True)
    canvas.text(102, 154, 125, 14, f"parliament · {hours(cohorts['hparl_train']['hours'])}", 11, font=MONO, color=BLUE_DARK)
    canvas.text(40, 202, 190, 14, 'spontaneous · fast · political', 11, font=MONO, color=CUE)
    canvas.rect(36, 234, 196, 84, fill=CARD, line='C2185B', line_width=1.5, radius=12, dash=True, name='CSS10 card')
    for x in (56, 76):  # open-book glyph
        canvas.rect(x, 258, 16, 22, line='C2185B', line_width=1.5, radius=3)
    canvas.text(102, 252, 120, 22, 'CSS10', 20, font=HEAD, color=NAVY, bold=True)
    canvas.text(102, 278, 125, 14, f"audiobook · {hours(cohorts['css10_train']['hours'])}", 11, font=MONO, color='C2185B')
    canvas.text(40, 326, 190, 14, 'read · literary · one voice', 11, font=MONO, color=CUE)
    canvas.line(232, 152, 300, 194, NAVY, 1.75, arrow=True)
    canvas.line(232, 276, 300, 220, 'C2185B', 1.75, dash=True, arrow=True)
    canvas.oval(262, 254, 13, 'C2185B')
    canvas.text(249, 243, 26, 22, '?', 17, font=HEAD, color='FFFFFF', bold=True, align='c')
    canvas.rect(304, 170, 150, 76, fill=NAVY, radius=12, name='Model')
    canvas.text(304, 188, 150, 22, 'Whisper-small', 18, font=HEAD, color=BG, bold=True, align='c')
    canvas.text(304, 215, 150, 14, 'fine-tune', 11, font=MONO, color='C9BFA8', align='c')
    canvas.line(454, 208, 500, 208, NAVY, 1.75, arrow=True)
    canvas.rect(504, 170, 146, 76, fill=CARD, line=BLUE_DARK, line_width=1.5, radius=12, name='Test')
    canvas.text(504, 188, 146, 22, 'Parliament test', 18, font=HEAD, color=NAVY, bold=True, align='c')
    canvas.text(504, 215, 146, 14, f"{cohorts['test']['utterances']:,} clips · frozen", 11, font=MONO, color=BLUE_DARK, align='c')


def slide_data(canvas, integrity):
    cohorts, cleaning = integrity['cohorts'], integrity['cleaning']
    total = cohorts['mixed_train']['hours'] + cohorts['dev']['hours'] + cohorts['test']['hours']
    canvas.header('02 · DATA', f'{total:.0f} hours — test frozen before cleaning')
    parts = [(cohorts['hparl_train']['hours'], BLUE_DARK), (cohorts['css10_train']['hours'], 'C2185B'),
             (cohorts['dev']['hours'], '9AA7BF'), (cohorts['test']['hours'], RED)]
    x, width = 28.0, 624.0
    edges = []
    for index, (value, color) in enumerate(parts):
        w = width * value / total
        canvas.rect(x, 104, w, 46, fill=color, radius=6 if index in (0, len(parts) - 1) else 0)
        edges.append((x, w))
        x += w
    (x0, w0), (x1, w1), (x2, w2), (x3, w3) = edges
    canvas.text(x0 + 16, 118, w0 - 20, 18, f"HParl train · {hours(cohorts['hparl_train']['hours'])}", 14, color='FFFFFF', bold=True)
    canvas.text(x1, 120, w1, 14, hours(cohorts['css10_train']['hours']), 11, font=MONO, color='FFFFFF', align='c')
    canvas.text(x2, 118, w2, 18, f"dev {hours(cohorts['dev']['hours'])}", 13, color=NAVY, bold=True, align='c')
    canvas.rect(x3 + 14, 124, 14, 11, fill='FFFFFF', radius=2)  # lock body
    canvas.rect(x3 + 17, 116, 8, 10, line='FFFFFF', line_width=1.5, radius=4)  # lock shackle
    canvas.text(x3 + 34, 118, w3 - 36, 18, f"test {hours(cohorts['test']['hours'])}", 13, color='FFFFFF', bold=True)
    train_right = x1 + w1
    canvas.line(28, 166, train_right, 166, MUTED, 0.75)
    canvas.line(28, 160, 28, 166, MUTED, 0.75)
    canvas.line(train_right, 160, train_right, 166, MUTED, 0.75)
    canvas.text(28, 172, train_right - 28, 14, f"train {hours(cohorts['mixed_train']['hours'])}", 11, font=MONO, color=MUTED, align='c')
    raw = cohorts['raw_hparl_train']['utterances']
    drops = cleaning['dropped_by_rule']
    steps = [(f'{raw:,}', 'raw clips', CARD, CARD_LINE, NAVY, '6B6457'),
             (f"−{drops['duplicate_audio']}", 'duplicate audio', 'FBEAF0', 'C2185B', 'C2185B', '8C1242'),
             (f"−{drops['cps_extreme']}", 'impossible speed', 'FBEAF0', 'C2185B', 'C2185B', '8C1242'),
             (f"{cohorts['hparl_train']['utterances']:,}", 'clean HParl', BLUE_DARK, None, 'FFFFFF', 'DFE6F2')]
    if raw - drops['duplicate_audio'] - drops['cps_extreme'] != cohorts['hparl_train']['utterances']:
        raise ValueError('cleaning counts do not add up')
    xs = [28, 188, 348, 508]
    for index, (number, label, fill, line, number_color, label_color) in enumerate(steps):
        w = 144 if index == 3 else 128
        canvas.rect(xs[index], 226, w, 58, fill=fill, line=line, radius=10)
        canvas.text(xs[index], 234, w, 22, number, 20, font=HEAD, color=number_color, bold=True, align='c')
        canvas.text(xs[index], 262, w, 14, label, 12, color=label_color, align='c')
        if index < 3:
            canvas.line(xs[index] + w + 4, 255, xs[index + 1] - 4, 255, MUTED, 1.25, arrow=True)
    canvas.text(28, 302, 400, 14, f"float32 · tags · {cleaning['review_queue_records']} flagged", 11, font=MONO, color=CUE)


def benchmark_rows(rows, extended):
    chosen = []
    for name, row in rows.items():
        source = extended.get(name, row)
        label, family = SYSTEMS[name]
        if name in extended and name == 'plain_ctc':
            label = 'Plain CTC'
        low, high = ci(source)
        chosen.append({'name': name, 'label': label, 'family': family, 'wer': source['wer_percent'], 'low': low,
                       'high': high, 'bold': '(ours)' in label})
    return sorted(chosen, key=lambda item: item['wer'])


def slide_benchmark(canvas, rows, extended):
    plot_rows = benchmark_rows(rows, extended)
    ours = min((r for r in plot_rows if r['family'] == 'contribution'), key=lambda r: r['wer'])
    official = [r for r in plot_rows if not rows[r['name']]['trained_by_group']]
    best_official = min(official, key=lambda r: r['wer'])
    if ours['wer'] >= min(r['wer'] for r in plot_rows if r['family'] != 'contribution'):
        raise ValueError('title claim no longer holds')
    canvas.header('03 · BENCHMARK', 'Our fine-tuned Whisper beats every baseline')
    axis_max = max(100, 5 * math.ceil(max(r['high'] for r in plot_rows) / 5))
    dot_plot(canvas, plot_rows, left=190, right=520, top=112, step=19, axis_max=axis_max,
             ticks=range(0, int(axis_max) + 1 - (int(axis_max) % 25), 25), label_size=12, dot_r=5.5)
    for index, (key, (label, color)) in enumerate(FAMILY.items()):
        canvas.oval(538, 116 + index * 22, 5, color)
        canvas.text(550, 108 + index * 22, 120, 16, label, 12, color='333333')
    canvas.text(532, 244, 130, 14, 'Best official:', 11, font=MONO, color=RED)
    canvas.text(532, 259, 130, 14, f"{best_official['label']} {best_official['wer']:.2f}", 11, font=MONO, color=RED)
    canvas.text(532, 282, 130, 14, f"Ours: {ours['wer']:.2f}", 11, font=MONO, color=RED)
    canvas.text(28, 362, 624, 14, 'dot = WER %, lower is better · bar = speaker-bootstrap 95% CI · '
                f"same {rows['canary']['records']:,} clips", 11, font=MONO, color=MUTED)


def slide_contribution(canvas, rows, extended, summary):
    pre, mixed, hparl = rows['whisper_pretrained'], rows['whisper_mixed'], rows['whisper_hparl']
    hubert, hubert_old, canary = extended['hubert_ctc'], rows['hubert_ctc'], rows['canary']
    canvas.header('04 · OUR CONTRIBUTION', f"Fine-tuning cuts Whisper's WER from {pre['wer_percent']:.0f}% to {mixed['wer_percent']:.1f}%")
    cards = [(28, 'Official Whisper-small', pre['wer_percent'], 'WER % · no training', CARD, CARD_LINE, '5F6B63', '1F8A5B', '1F8A5B'),
             (242, 'Our Whisper fine-tune', mixed['wer_percent'], 'HParl + CSS10 · best', 'FBEAF0', 'C2185B', '8C1242', 'C2185B', '8C1242'),
             (456, 'Our HuBERT-CTC head', hubert['wer_percent'], f"was {hubert_old['wer_percent']:.2f} · ≈ Canary",
              CARD, CARD_LINE, '5A3AA8', '7C4DDB', '5A3AA8')]
    for x, label, value, sub, fill, line, label_color, number_color, sub_color in cards:
        canvas.rect(x, 90, 196, 92, fill=fill, line=line, line_width=1.5 if line != CARD_LINE else 1, radius=10)
        canvas.text(x + 16, 100, 170, 16, label, 12, color=label_color, bold=x != 28)
        canvas.text(x + 16, 120, 170, 40, f'{value:.2f}', 38, font=HEAD, color=number_color, bold=True)
        canvas.text(x + 16, 163, 170, 14, sub, 11, font=MONO, color=sub_color)
    plot = []
    for row, label, family in ((mixed, 'Whisper FT +CSS10 (ours)', 'contribution'), (hparl, 'Whisper FT HParl (ours)', 'contribution'),
                               (canary, 'Canary-1b-v2', 'pretrained'), (hubert, 'mHuBERT-147 + CTC (ours)', 'ssl'),
                               (pre, 'Whisper-small (official)', 'pretrained')):
        low, high = ci(row)
        plot.append({'label': label, 'family': family, 'wer': row['wer_percent'], 'low': low, 'high': high, 'bold': '(ours)' in label})
    axis_max = 20 * math.ceil(max(r['high'] for r in plot) / 20)
    dot_plot(canvas, plot, left=200, right=560, top=226, step=23, axis_max=axis_max, ticks=range(0, axis_max + 1, 20),
             label_size=12, dot_r=6.5, annotate_from=480)
    comparison = summary['paired_data_composition_comparison']
    low, high = comparison['paired_speaker_bootstrap_95_ci']
    for index, line in enumerate(('+CSS10:', f"{comparison['delta_percentage_points']:.2f} pts".replace('-', '−'),
                                  f'CI [{low:.2f},'.replace('-', '−'), f'{high:.2f}]'.replace('-', '−'))):
        canvas.text(574, 218 + index * 15, 90, 14, line, 11, font=MONO, color=RED)
    canvas.text(28, 362, 624, 14, 'dot = WER %, lower is better · bar = speaker-bootstrap 95% CI · '
                f"{pre['records']:,} frozen test clips", 11, font=MONO, color=MUTED)


def slide_takeaways(canvas, example, integrity, summary):
    canvas.header('05 · TAKEAWAYS', 'Two lenses on error — and three limits')
    ref, hyp = example['reference'], example['prediction']
    split = next(i for i, (a, b) in enumerate(zip(ref, hyp)) if a != b)
    if ref[split + 1:] != hyp[split + 1:]:
        raise ValueError('example is expected to differ by one character')
    words, chars = example['word_counts'], example['character_counts']
    wer, cer = 100 * words['errors'] / words['reference_units'], 100 * chars['errors'] / chars['reference_units']
    canvas.rect(28, 90, 342, 208, fill=CARD, line=CARD_LINE, radius=12)
    for y, label, text, color in ((108, 'REF', ref, BLUE_DARK), (150, 'HYP', hyp, RED)):
        canvas.text(44, y + 3, 30, 14, label, 11, font=MONO, color=MUTED)
        canvas.text(80, y, 284, 20, [(text[:split], {}), (text[split], {'bold': True, 'color': color}), (text[split + 1:], {})],
                    15, font=GREEK, color=NAVY)
    canvas.text(80, 129, 284, 14, EXAMPLE_GLOSS, 11, font=MONO, color=CUE)
    for y, label, value, color in ((186, 'WER', wer, RED), (218, 'CER', cer, BLUE_DARK)):
        canvas.text(44, y - 1, 40, 16, label, 12, font=MONO, color=NAVY)
        canvas.rect(92, y, 200, 16, fill='E9E1D2', radius=8)
        canvas.rect(92, y, max(8.0, 200 * value / 100), 16, fill=color, radius=8 if value > 4 else 4)
        canvas.text(300, y, 64, 16, f'{value:.1f}%', 12, font=MONO, color=color)
    canvas.text(44, 252, 310, 14, 'ω = ο · inflection · corpus sums', 11, font=MONO, color=CUE)
    canvas.text(44, 274, 310, 14, 'real test clip · Whisper FT + CSS10', 11, font=MONO, color=MUTED)
    overlap = integrity['split_pair_group_overlap']['hparl_train__test']['speakers']
    test_speakers = integrity['cohorts']['test']['speakers_with_ids']
    delta = summary['paired_data_composition_comparison']['delta_percentage_points']
    tiles = [('1', ['training seed']), (f'{overlap}/{test_speakers}', ['test speakers', 'seen in train']),
             (f'{delta:.2f}'.replace('-', '−'), ['small CSS10 gain'])]
    for index, (number, label_lines) in enumerate(tiles):
        y = 90 + index * 73
        canvas.rect(386, y, 266, 62, fill=CARD, line=CARD_LINE, radius=12)
        canvas.text(404, y + 12, 90, 36, number, 30, font=HEAD, color=RED, bold=True)
        if len(label_lines) == 1:
            canvas.text(492, y + 22, 156, 18, label_lines[0], 14)
        else:
            canvas.text(492, y + 13, 156, 16, label_lines[0], 13)
            canvas.text(492, y + 31, 156, 16, label_lines[1], 13)
    canvas.rect(28, 318, 624, 40, fill=NAVY, radius=10)
    canvas.text(44, 330, 300, 16, 'Every number regenerates from code', 13, color=BG)
    canvas.text(352, 332, 284, 14, REPO_URL, 11, font=MONO, color='C9BFA8', align='r')


def speaker_notes(summary, rows, extended, integrity, example):
    """Short spoken cues per slide; every number comes from the same result files."""
    cohorts, cleaning = integrity['cohorts'], integrity['cleaning']
    total = cohorts['mixed_train']['hours'] + cohorts['dev']['hours'] + cohorts['test']['hours']
    drops = cleaning['dropped_by_rule']
    pre, hparl, mixed, canary = (rows[n]['wer_percent'] for n in ('whisper_pretrained', 'whisper_hparl', 'whisper_mixed', 'canary'))
    hubert = extended['hubert_ctc']['wer_percent']
    delta = -summary['paired_data_composition_comparison']['delta_percentage_points']
    words, chars = example['word_counts'], example['character_counts']
    overlap = integrity['split_pair_group_overlap']['hparl_train__test']['speakers']
    return [
        "Greek parliament speech: spontaneous, fast, political vocabulary.\n"
        "Question: does adding read audiobook speech (CSS10) help a fine-tuned Whisper?\n"
        "CSS10 is training-only; we always test on the original parliament test.",
        f"{total:.0f} hours in total. We froze the test set first, before any cleaning.\n"
        f"Cleaning fixed real problems: {drops['duplicate_audio']} duplicate recordings, {drops['cps_extreme']} clips with impossible "
        f"speech rate; CSS10 was stored as float32, so we fixed decoding; tags stripped; "
        f"{cleaning['review_queue_records']} suspicious clips flagged and logged.",
        f"All 10 required baselines plus our trained systems, same {cohorts['test']['utterances']:,} clips, one normalisation.\n"
        f"Colours are model families. Best official model: Canary, {canary:.1f}% WER.\n"
        f"Our fine-tuned Whisper: {mixed:.1f}%, the best result overall.",
        f"Official Whisper-small: {pre:.0f}%. Our fine-tuning on HParl: {hparl:.1f}%; adding CSS10: {mixed:.1f}%.\n"
        f"That is a {delta:.2f}-point gain, and its speaker-bootstrap interval stays below zero.\n"
        f"We also trained a CTC head on mHuBERT-147: {hubert:.1f}%, on par with Canary.",
        "A real test clip: one letter wrong, omega instead of omicron, the same sound.\n"
        f"WER counts the whole word: {100 * words['errors'] / words['reference_units']:.1f}%. "
        f"CER counts one character: {100 * chars['errors'] / chars['reference_units']:.1f}%. That is why we report both.\n"
        f"Limits: one seed; {overlap} of {cohorts['test']['speakers_with_ids']} test speakers also appear in training; "
        "the CSS10 gain is small.\nEvery number regenerates from the repository.",
    ]


def build_deck(data):
    summary, rows, extended, integrity, example = data
    deck = Presentation()
    deck.slide_width, deck.slide_height = Emu(int(SLIDE_W_IN * 914400)), Emu(int(7.5 * 914400))
    deck.core_properties.title = 'Greek Parliamentary ASR and Domain Adaptation'
    deck.core_properties.author = 'Anonymous group (COE025 Project 3)'
    blank = deck.slide_layouts[6]
    builders = [lambda c: slide_problem(c, integrity, rows), lambda c: slide_data(c, integrity),
                lambda c: slide_benchmark(c, rows, extended), lambda c: slide_contribution(c, rows, extended, summary),
                lambda c: slide_takeaways(c, example, integrity, summary)]
    for builder, notes in zip(builders, speaker_notes(summary, rows, extended, integrity, example)):
        slide = deck.slides.add_slide(blank)
        builder(Canvas(slide))
        slide.notes_slide.notes_text_frame.text = notes
    target = OUT / 'greek_parliamentary_asr.pptx'
    deck.save(target)
    embed_fonts(target)
    return target


# --------------------------------------------------------------------------- font embedding
EMBEDDED_FONTS = {  # typeface -> {style: vendored TrueType file}
    HEAD: {'bold': 'BricolageGrotesque-ExtraBold.ttf'},  # 800 weight in the Bold slot
    BODY: {'regular': 'Figtree-Regular.ttf', 'bold': 'Figtree-Bold.ttf'},
    MONO: {'regular': 'DMMono-Regular.ttf'},
}


def eot(path):
    """Wrap a TrueType file as uncompressed Embedded OpenType (v2.1), PowerPoint's .fntdata format."""
    import struct
    from fontTools.ttLib import TTFont

    data = path.read_bytes()
    font = TTFont(path)
    os2, head, names = font['OS/2'], font['head'], font['name']
    if os2.fsType & 0x0002:
        raise ValueError(f'{path.name} forbids embedding')
    def name(identifier):
        value = (names.getDebugName(identifier) or '').encode('utf-16-le')
        return struct.pack('<H', len(value)) + value
    panose = os2.panose
    body = struct.pack('<10B', panose.bFamilyType, panose.bSerifStyle, panose.bWeight, panose.bProportion, panose.bContrast,
                       panose.bStrokeVariation, panose.bArmStyle, panose.bLetterForm, panose.bMidline, panose.bXHeight)
    body += struct.pack('<BBIHH', 1, 1 if head.macStyle & 2 else 0, os2.usWeightClass, os2.fsType, 0x504C)
    body += struct.pack('<4I', os2.ulUnicodeRange1, os2.ulUnicodeRange2, os2.ulUnicodeRange3, os2.ulUnicodeRange4)
    body += struct.pack('<2I', getattr(os2, 'ulCodePageRange1', 1), getattr(os2, 'ulCodePageRange2', 0))
    body += struct.pack('<I4I', head.checkSumAdjustment, 0, 0, 0, 0)
    body += b'\0\0' + name(1) + b'\0\0' + name(2) + b'\0\0' + name(5) + b'\0\0' + name(4) + b'\0\0' + struct.pack('<H', 0)
    prefix = struct.pack('<III', len(data), 0x00020001, 0)  # FontDataSize, Version 2.1, Flags: no compression/XOR
    total = 4 + len(prefix) + len(body) + len(data)
    return struct.pack('<I', total) + prefix + body + data


def embed_fonts(target):
    """Embed the vendored fonts so the deck renders correctly on machines without them installed."""
    import shutil
    import tempfile
    import zipfile

    r_ns = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    font_rel = r_ns + '/font'
    with zipfile.ZipFile(target) as source:
        parts = {item.filename: source.read(item.filename) for item in source.infolist()}
    presentation = etree.fromstring(parts['ppt/presentation.xml'])
    rels = etree.fromstring(parts['ppt/_rels/presentation.xml.rels'])
    types = etree.fromstring(parts['[Content_Types].xml'])
    ct_ns = types.nsmap[None]
    if not any(d.get('Extension') == 'fntdata' for d in types):
        etree.SubElement(types, f'{{{ct_ns}}}Default', Extension='fntdata', ContentType='application/x-fontdata')
    used = {rel.get('Id') for rel in rels}
    font_list = etree.Element(qn('p:embeddedFontLst'))
    index = 0
    for typeface, styles in EMBEDDED_FONTS.items():
        entry = etree.SubElement(font_list, qn('p:embeddedFont'))
        etree.SubElement(entry, qn('p:font'), typeface=typeface)
        for style in ('regular', 'bold'):
            if style not in styles:
                continue
            index += 1
            rel_id = next(f'rIdFont{n}' for n in range(index, 10_000) if f'rIdFont{n}' not in used)
            used.add(rel_id)
            parts[f'ppt/fonts/font{index}.fntdata'] = eot(OUT / 'fonts' / styles[style])
            etree.SubElement(rels, f'{{{rels.nsmap[None]}}}Relationship', Id=rel_id, Type=font_rel,
                             Target=f'fonts/font{index}.fntdata')
            etree.SubElement(entry, qn(f'p:{style}')).set(f'{{{r_ns}}}id', rel_id)
    notes_size = presentation.find(qn('p:notesSz'))
    notes_size.addnext(font_list)
    presentation.set('embedTrueTypeFonts', '1')
    presentation.find(qn('p:sldSz')).attrib.pop('type', None)  # custom 16:9 size, not the template's 4:3 label
    parts['ppt/presentation.xml'] = etree.tostring(presentation, xml_declaration=True, encoding='UTF-8', standalone=True)
    parts['ppt/_rels/presentation.xml.rels'] = etree.tostring(rels, xml_declaration=True, encoding='UTF-8', standalone=True)
    parts['[Content_Types].xml'] = etree.tostring(types, xml_declaration=True, encoding='UTF-8', standalone=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix='.pptx') as handle:
        temporary = Path(handle.name)
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as output:
        output.writestr('[Content_Types].xml', parts.pop('[Content_Types].xml'))
        for name, content in parts.items():
            output.writestr(name, content)
    shutil.move(temporary, target)


# --------------------------------------------------------------------------- appendix PDF
def build_appendix(data):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    for path in sorted((OUT / 'fonts').glob('*.ttf')):
        font_manager.fontManager.addfont(str(path))
    matplotlib.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    summary, rows, extended, integrity, example = data
    color = lambda hex_value: '#' + hex_value

    def page(title, label):
        fig = plt.figure(figsize=(13.333, 7.5))
        fig.patch.set_facecolor(color(BG))
        fig.text(0.04, 0.93, label, fontfamily=MONO, fontsize=12, color=color(MUTED))
        fig.text(0.04, 0.855, title, fontfamily=HEAD, fontweight='bold', fontsize=30, color=color(NAVY))
        return fig

    def table(fig, columns, body, *, top=0.76, row_h=0.043, size=13, highlight=()):
        for x, heading, align in columns:
            fig.text(x, top, heading, fontfamily=MONO, fontsize=11, color=color(MUTED), ha='right' if align == 'r' else 'left')
        for index, values in enumerate(body):
            y = top - (index + 1) * row_h
            weight = 'bold' if index in highlight else 'normal'
            for (x, _, align), value in zip(columns, values):
                fig.text(x, y, value, fontfamily=[BODY, 'DejaVu Sans'], fontsize=size, color=color(INK), fontweight=weight,
                         ha='right' if align == 'r' else 'left')

    pages = []
    # A1 — full results table, all 14 rows.
    fig = page('Full results: 12 systems + 2 registered extended runs', 'APPENDIX A · ALL RESULTS ON 8,679 FROZEN TEST CLIPS')
    columns = [(0.04, 'SYSTEM', 'l'), (0.20, 'CHECKPOINT / IMPLEMENTATION', 'l'), (0.50, 'GROUP TRAINING', 'l'),
               (0.725, 'WER %', 'r'), (0.795, 'CER %', 'r'), (0.895, 'WER 95% CI', 'r'), (0.96, 'RTFx', 'r')]
    ordered = sorted(summary['rows'], key=lambda r: (r['role'] == 'contribution', r['wer_percent']))
    body, highlight = [], []
    for row in ordered:
        model = row['model_id'].replace('group-implemented/', 'own implementation: ')
        training = 'contribution: fine-tune' if row['role'] == 'contribution' else ('own training' if row['trained_by_group'] else 'none')
        body.append([row['system'], model, training, f"{row['wer_percent']:.2f}", f"{row['cer_percent']:.2f}",
                     f"[{row['wer_ci95_low_percent']:.2f}, {row['wer_ci95_high_percent']:.2f}]", f"{row['rtfx']:.1f}"])
        if row['role'] == 'contribution':
            highlight.append(len(body) - 1)
    for row in summary.get('extended_rows', []):
        if row['status'] == 'complete':
            body.append([row['system'], f"{rows[row['original_system']]['model_id'].replace('group-implemented/', 'own implementation: ')}",
                         f"own, {row['extended_updates']:,} updates", f"{row['wer_percent']:.2f}", f"{row['cer_percent']:.2f}",
                         f"[{row['wer_ci95_low_percent']:.2f}, {row['wer_ci95_high_percent']:.2f}]", f"{row['rtfx']:.1f}"])
    table(fig, columns, body, size=11.5, row_h=0.047, highlight=highlight)
    fig.text(0.04, 0.05, 'Source: results/benchmark/summary.json (scripts/summarize_benchmark.py). '
             'Extended rows: configs/extended_ctc_protocol.json.',
             fontfamily=MONO, fontsize=10, color=color(MUTED))
    pages.append(fig)
    # A2 — metrics.
    fig = page('Why WER and CER', 'APPENDIX B · METRICS AND SCORING POLICY')
    paragraphs = [
        ('WER = (S + D + I) / N words', 'Word substitutions, deletions and insertions over reference words: the standard ASR '
         'metric (Hugging Face audio course, ch. 5). It can exceed 100% when a system inserts words.'),
        ('CER = (S + D + I) / N characters', 'Character-level edits credit partly correct words. Greek is highly inflected and '
         'several letters share a sound (ω/ο, η/ι/υ), so one wrong letter makes a whole word wrong in WER.'),
        ('Corpus-level aggregation', 'Edit counts are summed over all 8,679 clips and then divided, instead of averaging '
         'per-utterance rates; short clips do not dominate.'),
        ('One frozen normalisation for every system', f"{integrity['normalization_policy']}: remove <tags>, NFC + casefold, "
         'punctuation to spaces, collapse whitespace. Accents, digits and symbols are kept. References and hypotheses alike.'),
        ('Uncertainty', '95% speaker-cluster bootstrap (1,000 resamples, seed 42) for every system, plus a paired bootstrap '
         'for the two matched Whisper fine-tuning conditions.'),
    ]
    for index, (head, text) in enumerate(paragraphs):
        y = 0.74 - index * 0.135
        fig.text(0.04, y, head, fontfamily=HEAD, fontweight='bold', fontsize=17, color=color(NAVY))
        fig.text(0.04, y - 0.025, text, fontfamily=[BODY, 'DejaVu Sans'], fontsize=13, color=color(INK), wrap=True, va='top')
    pages.append(fig)
    # A3 — split report required by the course.
    fig = page('Splits, speakers and overlap', 'APPENDIX C · DATA SPLIT REPORT')
    cohorts = integrity['cohorts']
    columns = [(0.04, 'SPLIT', 'l'), (0.36, 'SPEAKERS', 'r'), (0.50, 'UTTERANCES', 'r'), (0.62, 'HOURS', 'r')]
    body = [['HParl train (clean)', f"{cohorts['hparl_train']['speakers_with_ids']}", f"{cohorts['hparl_train']['utterances']:,}", f"{cohorts['hparl_train']['hours']:.2f}"],
            ['CSS10 train (single reader, no IDs)', '—', f"{cohorts['css10_train']['utterances']:,}", f"{cohorts['css10_train']['hours']:.2f}"],
            ['HParl dev', f"{cohorts['dev']['speakers_with_ids']}", f"{cohorts['dev']['utterances']:,}", f"{cohorts['dev']['hours']:.2f}"],
            ['HParl test (frozen)', f"{cohorts['test']['speakers_with_ids']}", f"{cohorts['test']['utterances']:,}", f"{cohorts['test']['hours']:.2f}"]]
    table(fig, columns, body, top=0.74, row_h=0.055, size=14)
    overlap = integrity['training_to_test_overlap']['hparl_train']
    facts = [f"Official HParl session split; it is NOT speaker-disjoint: {overlap['speakers']} test speakers also occur in training "
             f"({overlap['test_clips_with_shared_speaker']:,} test clips): not unseen-speaker generalisation.",
             f"Recording/session overlap between train and test: {overlap['recordings']}. Shared audio files: {overlap['shared_audio_sha256_values']}.",
             f"Test sentences found word for word in training: {overlap['literal_matching_test_utterances']} "
             f"(after the scoring normalisation: {overlap['final_policy_matching_test_utterances']}).",
             'CSS10 (literary audiobook) is used for training only and never enters dev or test.']
    for index, text in enumerate(facts):
        fig.text(0.04, 0.38 - index * 0.075, '•  ' + text, fontfamily=[BODY, 'DejaVu Sans'], fontsize=13, color=color(INK))
    fig.text(0.04, 0.05, 'Source: results/data_integrity.json (generated by scripts/report_benchmark_data.py).',
             fontfamily=MONO, fontsize=10, color=color(MUTED))
    pages.append(fig)
    # A4 — reproduction.
    fig = page('Reproduce every number', 'APPENDIX D · REPRODUCIBILITY')
    commands = ['python -m greek_asr.data verify-freeze', 'python -m greek_asr.data clean --selection configs/training_data_selection.json',
                'python scripts/run_benchmark_queue.py --phase test --output outputs/final_benchmark --nemo-python NEMO_PYTHON',
                'python scripts/summarize_benchmark.py', 'python presentation/build_presentation.py']
    for index, command in enumerate(commands):
        fig.text(0.04, 0.72 - index * 0.06, command, fontfamily=MONO, fontsize=13, color=color(NAVY))
    notes = ['Seed 42 everywhere; exact checkpoint revisions in configs/final_test_protocol.json.',
             'Test set frozen before cleaning; settings and checkpoints chosen on dev only; test decoded once per system.',
             'Pinned environments: requirements-*.txt; raw and cleaned data plus selected weights in the repository (Git LFS).',
             f'Repository: https://{REPO_URL}']
    for index, text in enumerate(notes):
        fig.text(0.04, 0.36 - index * 0.06, '•  ' + text, fontfamily=[BODY, 'DejaVu Sans'], fontsize=13, color=color(INK))
    pages.append(fig)
    target = OUT / 'appendix.pdf'
    with PdfPages(target, metadata={'Title': 'Greek Parliamentary ASR — appendix', 'Author': 'Anonymous group',
                                    'CreationDate': None, 'ModDate': None}) as pdf:
        for fig in pages:
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)
    return target


def main():
    data = load_data()
    print(json.dumps({'deck': str(build_deck(data).relative_to(ROOT)), 'appendix': str(build_appendix(data).relative_to(ROOT))}))


if __name__ == '__main__':
    main()
