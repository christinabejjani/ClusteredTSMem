    function shuffle(array) 
 	{
		var currentIndex = array.length, temporaryValue, randomIndex;

		// While there remain elements to shuffle...
		while (0 !== currentIndex) 
        {

			// Pick a remaining element...
			randomIndex = Math.floor(Math.random() * currentIndex);
			currentIndex -= 1;

			// And swap it with the current element.
			temporaryValue = array[currentIndex];
			array[currentIndex] = array[randomIndex];
			array[randomIndex] = temporaryValue;
		}

		return array;
	}

	var Stim1 = [];
    var Stim2 = [];
    var Stim3 = [];
    var Stim4 = []
    
    for (i=0; i < 30; i++)
    {
        Stim1=Stim1.concat(1);
        Stim2=Stim2.concat(2);
        Stim3=Stim3.concat(3);
        Stim4=Stim4.concat(4);
    }
	
    var StimSequence = [];
    StimSequence = StimSequence.concat(Stim3,Stim4,Stim1,Stim2);
    shuffle(StimSequence);
	console.log(StimSequence);