/*EvaluationKIT START*/
var evalkit_jshosted = document.createElement('script');
evalkit_jshosted.setAttribute('defer', 'defer');
evalkit_jshosted.setAttribute('type', 'text/javascript');
evalkit_jshosted.setAttribute('src', 'https://msoe.evaluationkit.com/canvas/js');
document.getElementsByTagName('head')[0].appendChild(evalkit_jshosted); 
/*EvaluationKIT END*/


/*Evercourse START*/
var partnerAccountShortcode = "cb_msoe_coursebuilder";
var hostServer = 'https://coursecontent.everspringpartners.com/evercourse';

var deploymentEnv = (function () {
    if (location.hostname.match(/test.instructure.com$/)) {
        return "test";
    } else if (location.hostname.match(/beta.instructure.com$/)) {
        return "beta";
    } else {
        return "prod";
    }
})();
var baseUrl = hostServer + "/canvas/" + deploymentEnv + "/" + partnerAccountShortcode;
$.getScript(baseUrl + "/evercourse.bundle.js").always(function () {
    $('[class*=esTemplate],[class*=eldpTemplate], #evercourse_dpr').fadeIn(500);
});
/*Evercourse END*/