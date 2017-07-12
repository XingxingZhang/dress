
from readability import Readability

def show_stat(text):
    rd = Readability(text)
    print 'Test text:'
    print '"%s"\n' % text
    print 'ARI: ', rd.ARI()
    print 'FleschReadingEase: ', rd.FleschReadingEase()
    print 'FleschKincaidGradeLevel: ', rd.FleschKincaidGradeLevel()
    print 'GunningFogIndex: ', rd.GunningFogIndex()
    print 'SMOGIndex: ', rd.SMOGIndex()
    print 'ColemanLiauIndex: ', rd.ColemanLiauIndex()
    print 'LIX: ', rd.LIX()
    print 'RIX: ', rd.RIX()

s1 = 'i saw him die .'
s2 = 'i watched him die .'
s1 = "it 's a classic ."
s2 = "this is a classic ."
s1 = "you 've lost weight ."
s2 = "you lost weight ."
s1 = "i am your mother ."
s2 = "i 'm your mother ."
s1 = "i told her ."
s2 = "i told him ."
s1 = "he is good ."
s2 = "she is good ."

show_stat(s1)

print '============================'

show_stat(s2)


