# Making the two Google Forms

The two forms are "Dance generation study" (Study 1, 12 videos with music) and "Dance style study" (Study 2,
15 silent videos). Each has a Group 1 and a Group 2 version. A participant picks the group you give them and sees
only that group's videos.

You need the 54 videos in `C:\Users\chinnu\3d_cvpr\form_studies`:
- `study1_disco\videos`: 24 files, `study1_v1_t01.mp4` ... `study1_v2_t12.mp4`;
- `study2_style\videos`: 30 files, `study2_v1_c01.mp4` ... `study2_v2_c15.mp4`.

Upload only the `.mp4` files. Never upload `answer_key.csv` or anything from `disco_study_pack`: they say which
model is A and which is B.

Use one Google account for everything (YouTube, the script, the forms). The forms and the response sheets will
belong to it.

## Step 1. Put the videos on YouTube (about 15 minutes)

Google Forms can only show videos that are on YouTube.

1. Open https://studio.youtube.com. If it asks you to create a channel, do so. A neutral channel name such as
   "Dance Study" is best, because participants see it under each video.
2. Click **Create → Upload videos**. Select up to 15 files at a time and upload all 54.
   **Do not rename them.** YouTube uses the file name as the title (`study1_v1_t01`), and the script finds each
   video by that title.
3. For each upload, set **Audience: "No, it's not made for kids"** and **Visibility: Unlisted**, never Private.
   Faster: close the upload dialogs, then on the **Content** page tick all the videos, choose **Edit →
   Visibility → Unlisted**, then **Edit → Audience → Not made for kids**, and click **Update videos**.
4. Wait until every video shows as processed, and check that there are 54.

## Step 2. Build the forms with the script (about 5 minutes)

1. Open https://script.google.com and click **New project**.
2. Delete the few lines already in the editor. Open
   `C:\Users\chinnu\3d_cvpr\DiscoForcing_code_2026-10-21\DiscoForcing\scripts\user_study\google_forms_builder.gs`
   in Notepad, copy all of it, and paste it into the editor.
3. On line 11, replace `YOUR NAME, your.email@example.com` with the contact shown to participants (for example
   your name and e-mail). Press **Ctrl+S**.
4. In the left sidebar, next to **Services**, click **+**, choose **YouTube Data API v3**, and click **Add**.
5. In the toolbar, make sure the function shown is **buildForms**, then click **Run**.
6. Google asks for permission:
   - click **Review permissions** and choose your account;
   - on "Google hasn't verified this app", click **Advanced → Go to Untitled project (unsafe)**. It is your own
     script, so this is expected;
   - click **Allow**.
7. Wait for "Execution completed" (1 to 3 minutes). The **Execution log** shows, for each form, an *edit* link,
   a *share* link and a *sheet* link. Keep them. Everything is also in a new Drive folder, "Dance user studies".

If it stops with "N videos not found on your YouTube channel: ...", those videos are missing, still processing, or
renamed. Fix them and click **Run** again. Each run makes new forms, so delete the half-made ones from Drive. As a
last resort, paste a link for each missing video into `LINKS` at the top of the script, e.g.
`'study2_v2_c15': 'https://youtu.be/AbCdEfGhIjK',`.

## Step 3. Check each form (2 minutes per form)

1. Open the *edit* link, then the **Settings** tab:
   - **Responses → Collect email addresses: Verified**;
   - **Limit to 1 response: on**;
   - **Presentation → Show progress bar: on**;
   - **Shuffle question order: off**.
2. Click the eye icon (**Preview**). Go through Group 1 to the end, then through Group 2. Check that the videos play
   (Study 1 with sound) and that the last video of a group goes straight to Submit.
3. Submit one test answer. Then go to **Responses → ⋮ → Delete all responses**, so your test is not in the data.
4. Click **Publish** (top right) when you are ready. Then use **Send → link icon** for the participant link (the
   *share* link from the log works too).

## Step 4. Send it to participants

- Aim for 24 to 30 people. Give them Group 1 and Group 2 alternately, in the order they agree to take part
  (1, 2, 1, 2, ...). Use the same group number for both studies.
- Do not invite co-authors or anyone who has seen the results.
- Message template:

  > Hi! Could you help with a short research study on computer-generated dance? There are two Google Forms:
  > Study 1 (about 12 minutes, needs sound): LINK1. Study 2 (about 7 minutes, silent): LINK2.
  > Please use a laptop or desktop and choose **Group N** when the form asks. You need to be signed in to a Google
  > account; the form records its e-mail only to prevent duplicate answers. Thank you!

- Responses appear in the two response sheets. When you are done, set each form to **not accepting responses**
  (Responses tab). Then download each sheet as CSV (**File → Download → CSV**) and send me both files. The analysis
  script reads them together with the private `answer_key.csv` files.

## If the script does not work: build a form by hand

Make one form per study. The steps are the same for both; the values in brackets are for Study 1 / Study 2.

1. Open https://forms.google.com and click **Blank form**. Title: *Dance generation study* / *Dance style study*.
   Paste the information text into the description. You can copy it from the `"intro"` entry in
   `google_forms_builder.gs`, replacing `{CONTACT}`.
2. **Settings**: Collect email addresses = Verified; Limit to 1 response = on; Show progress bar = on.
3. First question: a **Checkboxes** question titled *Consent* with one option, "I am 18 or older, I have read the
   information above, and I agree to take part." Make it **Required**.
4. **Add section** (the icon with two bars) titled *About you*. Add the profile questions as **Multiple choice**,
   all Required: dance experience; whether they can tell street styles apart; what they are watching on; and, for
   Study 1 only, whether the sound is on.
5. **Add section** titled *How to answer*. Paste the instructions (the `"instructions"` entry). Add a Multiple
   choice question *Which group number did the organiser give you?* with options *Group 1* and *Group 2*.
6. Build the first trial:
   - **Add section** titled *Group 1 - Trial 1 of 12* / *Group 1 - Clip 1 of 15*;
   - click **Add video** (the play icon), choose the **URL** tab and paste the link of `study1_v1_t01` /
     `study2_v1_c01` from YouTube Studio (on the video's ⋮ menu: **Get shareable link**);
   - add the questions, all Required. Study 1: three **Multiple choice** questions with options *A* and *B*
     (Style, Music, Naturalness, wording as in the script). Study 2: one **Linear scale** 1 to 5 labelled
     *Completely different style, or not dancing* and *Same style*, then a **Multiple choice** with *Yes*, *No*,
     *Not sure*;
   - end each question title with the trial's code, e.g. `[1-01]`, so every answer column has its own name.
7. For the other trials, use the section's **⋮ → Duplicate section**, then change only the section title, the
   video and the codes. Group 1 gets 12 / 15 sections, followed by the same number for Group 2 (`study1_v2_...`,
   codes `[2-01]` ...).
8. Navigation:
   - on the group question, **⋮ → Go to section based on answer**: Group 1 → *Group 1 - Trial 1*, Group 2 →
     *Group 2 - Trial 1*;
   - at the bottom of Group 1's last section, set **After section N → Submit form**.
9. **Responses → Link to Sheets**. Then do Step 3 above.

The video order inside each group must follow the file numbers (`t01`, `t02`, ...). That order was randomised once,
with the attention checks placed among the trials. Do not turn on "Shuffle question order".
