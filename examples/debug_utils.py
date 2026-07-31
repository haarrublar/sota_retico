### A debugging callback function that retico can send data to.
# here we use it as a way to see what the ASR is picking up. It's interesting and useful for debugging.
import retico_core

msg = []
def text_candidate_callback(update_msg):
    global msg

    for x, ut in update_msg:    #update_msg is  (IU, IU type)
        if ut == retico_core.UpdateType.ADD:
            msg.append(x)

        if ut == retico_core.UpdateType.REVOKE:
            if x in msg:
                msg.remove(x)

    # calculate the committed message so far
    txt = ""
    committed = False
    for x in msg:
        txt += x.text + " "
        committed = committed or x.committed

    if committed:
        msg = []
        print("\nCommitted: "+txt)

    else:
        print("\rlive: "+txt, end="")
