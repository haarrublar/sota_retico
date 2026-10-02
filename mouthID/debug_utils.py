import retico_core

ADD = retico_core.UpdateType.ADD
REVOKE = retico_core.UpdateType.REVOKE
COMMIT = retico_core.UpdateType.COMMIT


def make_text_callback(show=("live", "commit"), on_commit=None):
    show = set(show)
    msg = []

    def callback(update_msg):
        nonlocal msg
        committed = False

        for iu, ut in update_msg:
            if ut == ADD:
                msg.append(iu)
                if "add" in show:
                    print(f"\n[ADD]    {iu.text}")
            elif ut == REVOKE:
                if iu in msg:
                    msg.remove(iu)
                if "revoke" in show:
                    print(f"\n[REVOKE] {iu.text}")
            elif ut == COMMIT:
                committed = True

        txt = " ".join(iu.text for iu in msg)

        if committed:
            if "commit" in show:
                print(f"\n[COMMIT] {txt}")
            if on_commit:
                on_commit(txt)
            msg = []
        elif "live" in show:
            print(f"\r[LIVE]   {txt}", end="", flush=True)

    return callback
