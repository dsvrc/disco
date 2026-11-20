#!/usr/bin/env python3
"""
Write the Google Apps Script that builds both user-study forms in the organiser's Google account, from the
form_spec.json files of scripts/user_study/render_form_studies.py (no labels: which model is where stays in the
private answer_key.csv files).

  python scripts/user_study/make_forms_script.py --studies C:/Users/chinnu/3d_cvpr/form_studies \
      --out scripts/user_study/google_forms_builder.gs

Then follow scripts/user_study/GOOGLE_FORMS_HOWTO.md: upload the videos to YouTube (Unlisted, file names as titles),
paste the .gs file into a new Apps Script project, add the YouTube Data API service and run buildForms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COMMON_DATA = (
    "Your participation is voluntary. You can stop at any time by closing this page; nothing is sent to us until you "
    "press Submit. You must be 18 or older. No payment is offered.\n\n"
    "Your data: Google Forms records the e-mail address of the Google account you are signed in with. It is used only "
    "to make sure that each person answers once and to keep a record that the study took place. It is stored "
    "separately from your answers, is never published or shared, and is deleted when the research is published. Your "
    "answers are analysed and reported only anonymously and in aggregate, in a research publication.\n\n"
    "Questions about the study: {CONTACT}")

PROFILE = [
    {"title": "Your dance experience",
     "choices": ["None", "Some (casual, hobby or social dancing)",
                 "Trained (2 or more years of lessons, teaching or performing)"]},
    {"title": "Can you tell apart street-dance styles such as popping, locking or krump?",
     "choices": ["Yes", "No", "Not sure"]},
    {"title": "What are you using to watch the videos?", "choices": ["Laptop or desktop computer", "Tablet", "Phone"]}]

TEXT = {
    "disco": {
        "title": "Dance generation study",
        "intro": ("Thank you for helping with a research study on computer-generated dance.\n\n"
                  "What you will do: watch {N} short videos (15 seconds each, with music). Each video shows a real "
                  "dancer on the left and two computer-generated dances, A and B, made for the same music. For each "
                  "video you answer three quick questions. It takes about {MIN} minutes. Please use a laptop or "
                  "desktop computer with the sound on (headphones are best).\n\n" + COMMON_DATA),
        "profile_extra": [{"title": "Is your sound on?",
                           "choices": ["Yes, with headphones", "Yes, with speakers", "No"]}],
        "instructions": (
            "- Play each video with sound and watch it to the end. You can replay it.\n"
            "- Left: a real dancer (the reference). Right: A and B, two computer-generated dances for the same music.\n"
            "- Style: which of A or B dances more like the real dancer: the kind of steps, the arm and body "
            "movement, the energy.\n"
            "- Music: which of A or B moves better with the music: on the beat, and matching the music's energy.\n"
            "- Naturalness: which of A or B looks more like real human movement: fewer glitches, sliding feet or "
            "shaking.\n"
            "- Answer each question on its own. If A and B seem equal, choose the one you lean towards.\n"
            "- A few screens are attention checks with an obvious answer."),
        "page_help": "Play the video with sound and watch it to the end (you can replay it). Then answer the three "
                     "questions below it.",
    },
    "style": {
        "title": "Dance style study",
        "intro": ("Thank you for helping with a research study on dance styles.\n\n"
                  "What you will do: watch {N} short silent videos (10 seconds each). Each video shows two REFERENCE "
                  "dances in one style on the left and one TEST dance on the right. You rate how close the TEST "
                  "dance's style is to the references, and say whether the TEST figure is dancing. It takes about "
                  "{MIN} minutes. A laptop or desktop computer is best; no sound is needed.\n\n" + COMMON_DATA),
        "profile_extra": [],
        "instructions": (
            "- Play each video and watch it to the end. The TEST clip plays twice.\n"
            "- Judge the STYLE of the TEST dance against the two references: the kind of steps, the arm and "
            "upper-body movement, how energy and rhythm are used.\n"
            "- Do not judge smoothness, small glitches or how realistic the stick figure looks.\n"
            "- Rate from 1 (completely different style, or not dancing) to 5 (the same style).\n"
            "- Then say whether the TEST figure is dancing at all, in any style.\n"
            "- A few screens are checks with an obvious answer."),
        "page_help": "Play the video and watch it to the end (the TEST clip plays twice). Then answer the two "
                     "questions below it.",
    },
}

GS = r"""/**
 * Builds the two user-study forms ("Dance generation study" and "Dance style study") in your Google account.
 * Generated by scripts/user_study/make_forms_script.py: edit CONTACT below, nothing else.
 *
 * Before running (details in scripts/user_study/GOOGLE_FORMS_HOWTO.md):
 *   1. Upload every .mp4 of form_studies/study1_disco/videos and form_studies/study2_style/videos to YouTube as
 *      Unlisted, keeping the file names as titles (study1_v1_t01, ...).
 *   2. In this Apps Script project: Services (+) -> YouTube Data API v3 -> Add.
 * Then select buildForms in the toolbar and press Run. The log prints the links of both forms and response sheets.
 */
const CONTACT = 'YOUR NAME, your.email@example.com';  // shown on the first page of both forms
// Videos the YouTube lookup cannot find can be given by hand: 'study1_v1_t01': 'https://youtu.be/XXXXXXXXXXX',
const LINKS = {
};

const SPEC = __SPEC__;

function buildForms() {
  if (CONTACT.indexOf('example.com') >= 0) throw new Error('Edit CONTACT at the top of the script first.');
  const videos = findVideos_();
  const missing = [];
  Object.keys(SPEC).forEach(function (s) {
    Object.keys(SPEC[s].versions).forEach(function (v) {
      SPEC[s].versions[v].forEach(function (t) { if (!videos[t.video]) missing.push(t.video); });
    });
  });
  if (missing.length) {
    throw new Error(missing.length + ' videos not found on your YouTube channel: ' + missing.join(', ') +
                    '. Check that they are uploaded with these titles and finished processing, or add them to LINKS.');
  }
  const folder = DriveApp.createFolder('Dance user studies ' + Utilities.formatDate(new Date(), 'GMT', 'yyyy-MM-dd'));
  Object.keys(SPEC).forEach(function (s) {
    const r = buildForm_(SPEC[s], videos, folder);
    Logger.log(SPEC[s].title + '\n  edit:   ' + r.edit + '\n  share:  ' + r.share + '\n  sheet:  ' + r.sheet);
  });
  Logger.log('Both forms are in the Drive folder "' + folder.getName() + '". Open each edit link, check Settings ' +
             '(Responses: Collect email addresses = Verified, Limit to 1 response = on), preview it, then Publish.');
}

function findVideos_() {
  const map = {};
  Object.keys(LINKS).forEach(function (k) { map[k] = LINKS[k]; });
  let uploads;
  try {
    uploads = YouTube.Channels.list('contentDetails', {mine: true}).items[0].contentDetails.relatedPlaylists.uploads;
  } catch (e) {
    Logger.log('YouTube lookup unavailable (' + e + '): add the YouTube Data API v3 service, or fill LINKS.');
    return map;
  }
  let token = '';
  do {
    const res = YouTube.PlaylistItems.list('snippet', {playlistId: uploads, maxResults: 50, pageToken: token});
    (res.items || []).forEach(function (it) {
      const title = it.snippet.title.trim().replace(/\.mp4$/i, '');
      if (!map[title]) map[title] = 'https://www.youtube.com/watch?v=' + it.snippet.resourceId.videoId;
    });
    token = res.nextPageToken || '';
  } while (token);
  return map;
}

function pad_(n) { return (n < 10 ? '0' : '') + n; }

function buildForm_(S, videos, folder) {
  const form = FormApp.create(S.title);
  form.setTitle(S.title);
  form.setDescription(S.intro.replace('{CONTACT}', CONTACT));
  try {
    form.setEmailCollectionType(FormApp.EmailCollectionType.VERIFIED);
  } catch (e) {
    form.setCollectEmail(true);  // older Apps Script: check "Verified" in Settings by hand
  }
  form.setLimitOneResponsePerUser(true);
  form.setProgressBar(true);
  form.setShuffleQuestions(false);
  form.setAllowResponseEdits(false);
  form.setShowLinkToRespondAgain(false);
  form.setPublishingSummary(false);
  form.setConfirmationMessage('Thank you! Your answers have been recorded.');

  // page 1: information (form description) and consent
  form.addCheckboxItem().setTitle('Consent')
      .setChoiceValues(['I am 18 or older, I have read the information above, and I agree to take part.'])
      .setRequired(true);
  // page 2: about you
  form.addPageBreakItem().setTitle('About you');
  S.profile.forEach(function (q) {
    form.addMultipleChoiceItem().setTitle(q.title).setChoiceValues(q.choices).setRequired(true);
  });
  // page 3: instructions and group
  form.addPageBreakItem().setTitle('How to answer').setHelpText(S.instructions);
  const group = form.addMultipleChoiceItem().setTitle('Which group number did the organiser give you?')
      .setHelpText('If you were not given one, choose Group 1 if the day of your birthday is odd and Group 2 if it ' +
                   'is even.')
      .setRequired(true);
  // one page per video; each group's pages follow one another, the last page of a group submits
  const versions = Object.keys(S.versions).sort();
  const starts = {};
  versions.forEach(function (v, vi) {
    const trials = S.versions[v];
    trials.forEach(function (t, k) {
      const page = form.addPageBreakItem().setTitle('Group ' + v + ' - ' + t.title + ' of ' + trials.length)
          .setHelpText(S.page_help);
      if (k === 0) {
        starts[v] = page;
        if (vi > 0) page.setGoToPage(FormApp.PageNavigationType.SUBMIT);  // after the previous group's last page
      }
      form.addVideoItem().setTitle(t.title).setVideoUrl(videos[t.video]).setWidth(640)
          .setAlignment(FormApp.Alignment.CENTER);
      const code = ' [' + v + '-' + pad_(t.position) + ']';  // makes every response column unique
      S.questions.forEach(function (q) {
        if (q.scale) {
          form.addScaleItem().setTitle(q.text + code).setBounds(q.scale[0], q.scale[1]).setLabels(q.low, q.high)
              .setRequired(true);
        } else {
          form.addMultipleChoiceItem().setTitle(q.text + code).setChoiceValues(q.choices).setRequired(true);
        }
      });
    });
  });
  group.setChoices(versions.map(function (v) { return group.createChoice('Group ' + v, starts[v]); }));

  const sheet = SpreadsheetApp.create(S.title + ' (responses)');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, sheet.getId());
  DriveApp.getFileById(form.getId()).moveTo(folder);
  DriveApp.getFileById(sheet.getId()).moveTo(folder);
  return {edit: form.getEditUrl(), share: form.getPublishedUrl(), sheet: sheet.getUrl()};
}
"""


def spec_of(path: Path, kind: str) -> dict:
    s = json.loads(path.read_text())
    txt = TEXT[kind]
    n = max(len(v) for v in s["versions"].values())
    versions = {v: [{"position": t["position"], "title": t["title"], "video": Path(t["video"]).stem} for t in ts]
                for v, ts in s["versions"].items()}
    return {"title": txt["title"],
            "intro": txt["intro"].replace("{N}", str(n)).replace("{MIN}", str(s.get("minutes", 10))),
            "profile": PROFILE + txt["profile_extra"], "instructions": txt["instructions"],
            "page_help": txt["page_help"], "questions": s["questions"], "versions": versions}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studies", type=Path, default=Path("form_studies"), help="render_form_studies.py --out folder")
    ap.add_argument("--out", type=Path, default=Path("scripts/user_study/google_forms_builder.gs"))
    a = ap.parse_args()
    spec = {"study1": spec_of(a.studies / "study1_disco" / "form_spec.json", "disco"),
            "study2": spec_of(a.studies / "study2_style" / "form_spec.json", "style")}
    a.out.write_bytes(GS.replace("__SPEC__", json.dumps(spec, indent=1, ensure_ascii=False)).encode("utf-8"))
    n = sum(len(t) for s in spec.values() for t in s["versions"].values())
    print(f"wrote {a.out}: {n} video pages in 2 forms; videos to upload: "
          + ", ".join(sorted({t['video'] for s in spec.values() for ts in s['versions'].values() for t in ts}))[:200]
          + " ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
