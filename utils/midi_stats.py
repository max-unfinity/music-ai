import numpy as np
import pretty_midi
from scipy.stats import entropy
from sklearn.cluster import DBSCAN


minor_scale = np.array([0,2,3,5,7,8,10])
major_scale = np.array([0,2,4,5,7,9,11])

minor_scale_mask = np.zeros(12, dtype=bool)
minor_scale_mask[minor_scale] = True
minor_scale_strength_idxs = minor_scale[np.diff(minor_scale, prepend=-2) == 2]

major_scale_mask = np.zeros(12, dtype=bool)
major_scale_mask[major_scale] = True
major_scale_strength_idxs = major_scale[np.diff(major_scale, prepend=-2) == 2]


def get_midi_stats(midi, note_duration_step=0.1, note_duration_k=2, tempo_step=10., note_velocity_step=20.):
    # tempo
    t1, t2 = midi.get_tempo_changes()
    t1 = np.diff(t1, append=midi.get_end_time())
    p = t1/t1.sum()
    tempo_mean = (t2*p).sum()

    i_sort = np.argsort(t2)
    bins = np.round(t2[i_sort]/tempo_step)*tempo_step
    u, splits = np.unique(bins, return_index=True)
    splits = splits[1:]
    bins2 = np.split(bins, splits)
    p2 = np.split(p[i_sort], splits)
    y = [x.sum() for x in p2]
    tempo_hist = u, np.array(y)

    # time_signature
    t1 = [x.time for x in midi.time_signature_changes]
    t1 = np.diff(t1, append=midi.get_end_time())
    p = t1/t1.sum()
    time_signature_hist = {}
    for x,p_i in zip(midi.time_signature_changes, p):
        ts = f'{x.numerator}/{x.denominator}'
        if not time_signature_hist.get(ts):
            time_signature_hist[ts] = 0
        time_signature_hist[ts] += p_i
        
    # note-based stats
    all_notes = [x for inst in midi.instruments for x in inst.notes if not inst.is_drum]
    if len(all_notes) == 0:
        return None
    all_notes_np = np.array([[x.start,x.end,x.pitch,x.velocity] for x in all_notes])
    all_notes_np_filtered = filter_duplicate_notes(all_notes_np)
    
    track_duration = midi.get_end_time()
    has_drums = bool(max([inst.is_drum for inst in midi.instruments]))
    n_notes = len(all_notes_np_filtered)
    n_notes_raw = len(all_notes)
    n_bars = len(midi.get_downbeats())
    density = n_notes/track_duration

    durations = all_notes_np_filtered[:,1] - all_notes_np_filtered[:,0]
    note_duration_stats = durations.mean(), durations.std()
    i, c = get_hist_log(durations, note_duration_step, note_duration_k)
    c = c/n_notes
    note_duration_hist = i, c

    velocities = all_notes_np_filtered[:,3]
    note_velocity_stats = velocities.mean(), velocities.std()
    i, c = get_hist(velocities, note_velocity_step)
    c = c/n_notes
    note_velocity_hist = i, c

    pitches = all_notes_np_filtered[:,2].astype(int)
    pitch_diversity = np.zeros(12)
    i, c = np.unique(pitches%12, return_counts=True)
    pitch_diversity[i] = c/n_notes

    pitch_hist_128 = np.zeros(128)
    i, c = np.unique(pitches, return_counts=True)
    pitch_hist_128[i] = c/n_notes
    
    octave_diversity = np.zeros(11)
    i, c = np.unique(pitches.astype(int)//12, return_counts=True)
    octave_diversity[i] = c/n_notes
    
    pitch_entropy = entropy(pitch_diversity, base=12)
    octave_entropy = entropy(octave_diversity, base=11)
    pitch_128_entropy = entropy(pitch_hist_128, base=128)
    
    # advanced stats
    start2start, end2start = get_starts_ends(all_notes_np_filtered)
    
    note_repetition_2 = detect_note_repetition_2(all_notes_np_filtered)
    
    tonic, scale_type, tonic_loss, d_normed, attempt = estimate_tonic(pitch_diversity)
    
    if len(start2start):
        x,y,loss = get_clustered_duration_hist(start2start, end2start)
        clustered_duration_hist = [x,y]
        clustered_duration_loss = loss
        sparsity, broadness = get_sparisty_broadness(all_notes_np_filtered, start2start, end2start, track_duration)
    else:
        clustered_duration_hist, clustered_duration_loss = None,None
        sparsity, broadness = None,None
    
    
    return dict(
        tempo_mean=tempo_mean,
        tempo_hist=tempo_hist,
        time_signature_hist=time_signature_hist,
        track_duration=track_duration,
        has_drums=has_drums,
        n_notes=n_notes,
        n_notes_raw=n_notes_raw,
        n_bars=n_bars,
        density=density,
        note_duration_stats=note_duration_stats,
        note_duration_hist=note_duration_hist,
        note_velocity_stats=note_velocity_stats,
        note_velocity_hist=note_velocity_hist,
        pitch_diversity=pitch_diversity,
        octave_diversity=octave_diversity,
        pitch_hist_128=pitch_hist_128,
        pitch_entropy=pitch_entropy,
        octave_entropy=octave_entropy,
        pitch_128_entropy=pitch_128_entropy,
        note_repetition_2=note_repetition_2,
        tonic_estimation=[tonic, scale_type, attempt],
        tonic_loss=tonic_loss,
        clustered_duration_hist=clustered_duration_hist,
        clustered_duration_loss=clustered_duration_loss,
        sparsity=sparsity,
        broadness=broadness,
    )


def get_starts_ends(notes, return_sort=False):
    i_sort_start = np.argsort(notes[:,0])
    i_sort_end = np.argsort(notes[:,1])
    end2start = notes[i_sort_start,0][1:] - notes[i_sort_end,1][:-1]
    start2start = np.diff(notes[i_sort_start,0])
    if return_sort:
        return start2start, end2start, i_sort_start, i_sort_end
    else:
        return start2start, end2start


def get_sparisty_broadness(notes, start2start, end2start, track_duration):
    pauses = end2start[end2start>0].sum()
    sparsity = pauses/track_duration
    broadness = (notes[:,1]-notes[:,0]).sum()/(track_duration-pauses)
    return sparsity, broadness


def detect_note_repetition_2(notes, thershold_len=0.14, threshold_sec=0.01):   
    i_sort = np.lexsort([notes[:,0], notes[:,2]])
    s = []
    lens = []
    cur_p = -1
    cur_t = -1
    for t in notes[i_sort]:
        a,b,p,v = t
        if cur_p != p:
            cur_p = p
        else:
            s.append(a-cur_t)
            lens.append(b-a)
        cur_t = b
    s = np.array(s)
    lens = np.array(lens)
    return ((s < threshold_sec) & (lens <= thershold_len)).sum()/len(s)


def estimate_tonic(pitch_diversity):
    """Returns: tonic, scale_type, tonic_loss, d_normed, attempt"""
    d = pitch_diversity
    i_sort = d.argsort()
    scale_type = -1
    tonic_max = None
    tonics = []
    losses = []
    for attempt in range(5):
        tonic = i_sort[-attempt-1]
        if tonic_max is None:
            tonic_max = tonic
        elif d[tonic]/d[tonic_max] < 0.7:
            break
        d_normed = np.roll(d, -tonic)
        diff = np.diff(d_normed, prepend=0)

        loss1 = -np.minimum(diff[minor_scale_strength_idxs], 0)
        loss2 = np.maximum(diff[~minor_scale_mask], 0)
        loss_minor = np.sum(loss1) + np.sum(loss2)

        loss1 = -np.minimum(diff[major_scale_strength_idxs], 0)
        loss2 = np.maximum(diff[~major_scale_mask], 0)
        loss_major = np.sum(loss1) + np.sum(loss2)

        err_minor = d_normed[~minor_scale_mask].sum()
        err_major = d_normed[~major_scale_mask].sum()
        losses.append([loss_minor+err_minor,0,tonic])
        losses.append([loss_major+err_major,1,tonic])

        if loss_minor == 0:
            scale_type = 0
            break
        if loss_major == 0:
            scale_type = 1
            break
    best = np.argsort(np.array(losses)[:,0], kind='stable')[0]
    loss, scale_type, tonic = losses[best]
    d_normed = np.roll(d, -tonic)
    return tonic, scale_type, loss, d_normed, attempt


def get_clustered_duration_hist(start2start, end2start, eps=0.02):
    dbscan = DBSCAN(eps=eps, min_samples=1)
    labels = dbscan.fit_predict(start2start[:,None])
    
    clusters = np.unique(labels)
    loss = []
    for c in clusters:
        x = start2start[labels==c]
        l = np.max(np.abs(x - x.mean()))
        loss.append(l)
    loss = np.mean(loss)

    clusters, counts = np.unique(labels, return_counts=True)
    clustered = []
    for c in clusters:
        clustered.append(start2start[labels==c])
    clustered = sorted(clustered, key=lambda x: len(x), reverse=True)
    n = len(start2start)
    x = np.array([np.median(x) for x in clustered])
    y = np.array([len(x)/n for x in clustered])
    m = (y > 0.01) & (x > 0)
    x, y = x[m], y[m]
    return x, y, loss


def detect_pauses(start2start, end2start, silence_threshold_sec=0.1, sum_threshold_sec=3, return_scores=False):
    idxs = np.where((end2start >= silence_threshold_sec)
                    & (start2start + end2start >= sum_threshold_sec))[0]
    if return_scores:
        scores = np.clip(end2start, 0, None)*2 + start2start
        return idxs+1, scores
    return idxs+1


def filter_duplicate_notes(notes):
    v,idxs = np.unique(notes[:,:3], axis=0, return_index=True)  # filter duplicates, ignoring velocity
    notes_filtered = notes[idxs]
    return notes_filtered

def get_hist(x, step):
    x = np.round(x/step)*step
    return np.unique(x, return_counts=True)

def get_hist_log(x, step, k):
    x = (np.exp(np.round(np.log(x*k+1)/step)*step)-1)/k
    return np.unique(x, return_counts=True)

def get_all_notes_np(midi):
    all_notes = [x for inst in midi.instruments for x in inst.notes if not inst.is_drum]
    if len(all_notes) == 0:
        return None
    all_notes_np = np.array([[x.start,x.end,x.pitch,x.velocity] for x in all_notes])
    return all_notes_np
