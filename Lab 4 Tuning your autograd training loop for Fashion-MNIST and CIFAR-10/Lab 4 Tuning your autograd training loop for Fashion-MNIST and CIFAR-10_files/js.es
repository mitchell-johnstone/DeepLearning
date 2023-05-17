var evalkit_setup = { auth_external_tool_id: 436, account_url: 'https://msoe.evaluationkit.com', account_id: 1, deployment_id: null, auth_url: '/api/v1/accounts/1/external_tools/sessionless_launch?id=436&url=https%3a%2f%2fmsoe.evaluationkit.com%2fCanvas%2fLti' };var evalkit_loaded=0,EvaluationKIT={onLoad:function(){var e,t;null===ENV.current_user_id||-1<window.location.href.indexOf("sessionless_launch")||-1<window.location.href.indexOf("display=borderless")||0===$("#dashboard").length&&0===$(".context_external_tool_"+evalkit_setup.auth_external_tool_id).length&&0===this.getPageCourseId()&&-1===document.location.href.indexOf("/profile")&&-1===document.location.href.indexOf("/grades")||3<(evalkit_loaded+=1)||(e=this.getTokenCookie(),t=this.getVerifyCookie(),0===e.length||t!==ENV.current_user_id?this.onLti():this.onUser(e))},onLti:function(){this.startInit(function(){(EvaluationKIT||{}).onAuth()})},startInit:function(n){var o=this,e=o.getInitCookie();null!==e&&!1===e||(null===e||!0!==e?$.get("/api/v1/users/self/profile",function(t){var e;0<t.login_id.length?0<(e=o.getPageCourseId())?$.get("/api/v1/courses/"+e,function(e){0<e.id?o.onAuthInit(t.id,t.login_id,t.integration_id,e.id,e.course_code,n):o.setInitCookie(!1)}):o.onAuthInit(t.id,t.login_id,t.integration_id,0,"",n):o.setInitCookie(!1)}):n())},onAuthInit:function(e,t,n,o,i,a){var l=this;$.get(evalkit_setup.account_url+"/Canvas/InitCheck?authId="+evalkit_setup.auth_external_tool_id+"&username="+encodeURIComponent(t)+"&userid="+e+"&integration_id="+encodeURIComponent(n)+"&coursecode="+encodeURIComponent(i)+"&courseid="+o+"&refer="+encodeURIComponent(window.location.pathname+window.location.search)+"&nocache="+(new Date).getTime(),function(e){l.setInitCookie(e.result),!1!==e.result&&a()})},onAuth:function(){var t;null===evalkit_setup.deployment_id?(t=this.createCORSRequest("GET",evalkit_setup.auth_url))&&(t.onload=function(){var e;200===t.status?(e=$.parseJSON(t.responseText.replace("while(1);","")),0<$("#evalkit-lti").length&&$("#evalkit-lti").remove(),$('<iframe src="'+e.url+'" id="evalkit-lti" style="display:none;"></iframe>').appendTo($("body"))):(EvaluationKIT||{}).onLoad()},t.send()):this.onAuth3()},onAuth3:function(){0<$("#evalkit-lti").length&&$("#evalkit-lti").remove(),$('<iframe src="'+evalkit_setup.auth_url+'" id="evalkit-lti" style="display:none;"></iframe>').appendTo($("body"))},onLtiResponse:function(e){if(null!==e.origin&&-1!==e.origin.toLowerCase().indexOf(evalkit_setup.account_url)&&null!==e.data&&void 0!==e.data.length){var t=EvaluationKIT||{},n=JSON.parse(e.data);if(null!==n)switch(n.subject){case"evalkit.location":window.top.location.href=n.url;break;case"evalkit.ltiresponse":var o=n.token;0===o.length&&(o="-1"),t.setTokenCookie(o),t.setVerifyCookie(ENV.current_user_id),t.onLoad()}else console.log("DEBUG: onLtiResponse - jsondata error")}},onUser:function(o){"-1"!==o&&this.startInit(function(){var e,t=EvaluationKIT||{},n=t.getPageCourseId();null===document.getElementById("ek_css")&&((e=document.createElement("link")).setAttribute("id","ek_css"),e.setAttribute("rel","stylesheet"),e.setAttribute("type","text/css"),e.setAttribute("href",evalkit_setup.account_url+"/scripts/canvas/style.min.css"),document.getElementsByTagName("head")[0].appendChild(e)),0===n?t.getUserSettings(o,n,""):$.ajax({url:"/api/v1/courses/"+n,type:"GET",success:function(e){var t="";try{t=e.course_code}catch(e){(EvaluationKIT||{}).onError(e)}(EvaluationKIT||{}).getUserSettings(o,n,t)},error:function(e){var t=EvaluationKIT||{};t.onError(e),t.onLoad()}})})},getUserSettings:function(e,n,t){var o=this.createCORSRequest("GET",evalkit_setup.account_url+"/canvas/usersettings?token="+encodeURIComponent(e)+"&userid="+ENV.current_user_id+(0<n?"&courseid="+n+"&coursecode="+encodeURIComponent(t)+"&courseTitle=":"")+"&refer="+encodeURIComponent(window.location.pathname+window.location.search)+"&nocache="+(new Date).getTime());o&&(o.onload=function(){if(401===o.status)(EvaluationKIT||{}).onLti();else try{var e=$.parseJSON(o.responseText),t=EvaluationKIT||{};t.onPopup(e),t.onWidget(e),t.onLinks(e,n),$(document).on("click",".ek-sessionless",function(e){var t,n=$(this).attr("href");e.preventDefault(),null===evalkit_setup.deployment_id?(t=(EvaluationKIT||{}).createCORSRequest("GET",n))&&(t.onload=function(){var e;200===t.status&&void 0!==(e=$.parseJSON(t.responseText.replace("while(1);",""))).url&&null!==e&&0<e.url.length?ek_modal.open({iframelink:e.url,buttons:null,iframeheight:800,cssclass:"evalkit-modal-lg",showClose:!0,type:"notification"}):(EvaluationKIT||{}).onLti()},t.send()):ek_modal.open({iframelink:n,buttons:null,iframeheight:800,cssclass:"evalkit-modal-lg",showClose:!0,type:"notification"})})}catch(e){(EvaluationKIT||{}).onError(e)}},o.onerror=function(e){var t=EvaluationKIT||{};-1<navigator.appVersion.indexOf("MSIE 10")||-1<navigator.appVersion.indexOf("MSIE 9")?t.onLti():t.onError(e)},o.send())},onPopup:function(t){var e,n;!0===t.popup.visible&&(e=[],(n=document.createElement("a")).innerHTML=t.popup.gotosurveytext,-1<t.popup.gotosurveyurl.indexOf("sessionless_launch")||-1<t.popup.gotosurveyurl.indexOf("display=borderless")?n.setAttribute("class","Button Button--primary ek-widget-btn-primary ek-sessionless"):n.setAttribute("class","Button Button--primary ek-widget-btn-primary"),n.setAttribute("href",t.popup.gotosurveyurl),e[0]=n,t.popup.remindlater&&((n=document.createElement("a")).innerHTML=t.popup.remindlatertext,n.setAttribute("href","#"),n.setAttribute("class","Button ek-widget-btn-default"),!0===t.popup.incrementDefer?n.onclick=function(e){e.preventDefault(),ek_modal.close();e=EvaluationKIT||{};$.get(evalkit_setup.account_url+"/canvas/defer?token="+encodeURIComponent(e.getTokenCookie())+"&projectId="+t.popup.projectId+"&courseId="+t.popup.courseId+"&nocache="+(new Date).getTime())}:n.onclick=function(e){e.preventDefault(),ek_modal.close()},e[1]=n),ek_modal.open({title:t.popup.header,body:t.popup.body,buttons:e,showClose:!1,type:!0===t.popup.blockPage?"blocker":"notification"}))},onWidget:function(e){setTimeout(function(){$(e.widget).appendTo("#right-side")},2e3)},onLinks:function(e,t){this.setLink(e.userlink,t),this.setLink(e.studentlink,t),this.setLink(e.instructorlink,t),this.setLink(e.talink,t),this.setLink(e.adminlink,t)},createCORSRequest:function(e,t){try{var n=new XMLHttpRequest;return"withCredentials"in n?n.open(e,t,!0):"undefined"!=typeof XDomainRequest?(n=new XDomainRequest).open(e,t):n=null,n}catch(e){this.onError(e)}},getPageCourseId:function(){return void 0!==$("body").attr("class")&&$("body").attr("class").match(/\bcontext-course_(.[0-9]*)/)?parseInt($("body").attr("class").match(/\bcontext-course_(.[0-9]*)/)[1]):0},getPageUserId:function(){return $("body").attr("class").match(/\bcontext-user_(.[0-9]*)/)?$("body").attr("class").match(/\bcontext-user_(.[0-9]*)/)[1]:""},getTokenCookie:function(){return evalkit_readCookie("evalkit_token_"+evalkit_setup.auth_external_tool_id)||""},setTokenCookie:function(e){evalkit_createCookie("evalkit_token_"+evalkit_setup.auth_external_tool_id,e)},getVerifyCookie:function(){return evalkit_readCookie("evalkit_verify_"+evalkit_setup.auth_external_tool_id)||""},setVerifyCookie:function(e){evalkit_createCookie("evalkit_verify_"+evalkit_setup.auth_external_tool_id,e)},getInitCookie:function(){var e=evalkit_readCookie("evalkit_init_verfiy_"+evalkit_setup.auth_external_tool_id);if(null!==e&&""!==e||(e=-1),-1!==e&&e!==ENV.current_user_id)return evalkit_createCookie("evalkit_init_verfiy_"+evalkit_setup.auth_external_tool_id,ENV.current_user_id),evalkit_createCookie("evalkit_init_"+evalkit_setup.auth_external_tool_id,""),null;e=evalkit_readCookie("evalkit_init_"+evalkit_setup.auth_external_tool_id);if(null===e||""===e)return null;var t=this.getSection();if(null===t)return!1;for(var n=JSON.parse(e),o=0;o<n.length;o++)if(n[o].section===t)return n[o].val;return null},setInitCookie:function(e){var t=evalkit_readCookie("evalkit_init_"+evalkit_setup.auth_external_tool_id);null!==t&&""!==t||(t="");var n=this.getSection();if(null!==n){for(var o=0===t.length?[]:JSON.parse(t),i=!1,a=0;a<o.length;a++)o[a].section===n&&(i=!0,o[a].val=e);!1===i&&o.push({section:n,val:e}),evalkit_createCookie("evalkit_init_verfiy_"+evalkit_setup.auth_external_tool_id,ENV.current_user_id),evalkit_createCookie("evalkit_init_"+evalkit_setup.auth_external_tool_id,JSON.stringify(o))}},getSection:function(){var e=null;return 0<$("#dashboard").length?e="dashboard":-1<document.location.href.indexOf("/profile")?e="profile":0<this.getPageCourseId()?e=this.getPageCourseId():-1<document.location.href.indexOf("/grades")&&(e="dashboard"),e},setLink:function(e){var t;null!==e&&null!==e.url&&(t=$("a.context_external_tool_"+e.exttoolid),e.visible?(0===t.length||t.hasClass("evalkit-ltilink")?($("<li class='section evalkitlink'><a style='display:block !important;' class='evalkit-ltilink ltilinkcontext_external_tool_"+e.exttoolid+"' href='"+e.url+"'>"+e.title+"</a></li>").appendTo("#section-tabs"),t=$("a.context_external_tool_"+e.exttoolid)):(t.attr("style","display: block !important;"),t.attr("href",e.url),e.title!==t.html()&&t.html(e.title)),null!==e.badge&&$('<b class="nav-badge evalkit-badge">'+e.badge+"</b>").prependTo(t),t.parent().show(),t.addClass("evalkit-ltilink")):t.hasClass("evalkit-ltilink")||t.parent().hide())},onError:function(e){}};$(document).ready(function(){var e=EvaluationKIT||{};window.addEventListener?window.addEventListener("message",e.onLtiResponse,!1):window.attachEvent("onmessage",e.onLtiResponse),e.onLoad()});var ek_modal=function(){var o,i,a,l,r,e={},s=document.createElement("div");s.id="ek-overlay",s.style.display="none",(o=document.createElement("div")).id="ek-modal",o.style.display="none",o.setAttribute("role","dialog"),o.setAttribute("tabindex","-1"),o.setAttribute("aria-live","assertive"),o.setAttribute("aria-labelledby","ek-modal-header");var t=document.createElement("div");return t.setAttribute("class","ek-modal-content"),(i=document.createElement("h2")).setAttribute("class","ek-modal-header"),i.id="ek-modal-header",(r=document.createElement("a")).setAttribute("id","ek-modal-close"),(a=document.createElement("div")).setAttribute("class","ek-modal-body"),a.setAttribute("id","ek-modal-body"),(l=document.createElement("div")).setAttribute("class","ek-modal-footer"),t.appendChild(i),t.appendChild(r),t.appendChild(a),t.appendChild(l),o.appendChild(t),e.open=function(e){if(null!==e){var t;if(void 0!==e.cssclass&&null!==e.cssclass&&o.setAttribute("class",e.cssclass),a.innerHTML="",void 0!==e.iframelink&&null!==e.iframelink&&0<e.iframelink.length?(a.innerHTML='<iframe id="ek-modal-iframe-loading"></iframe><iframe src='+e.iframelink+' id="ek-modal-iframe" style=\'display:none;\' onload="evalkit_modal_iframe();"></frame>',window.removeEventListener("resize",evalkit_modal_iframe_resize),window.addEventListener("resize",evalkit_modal_iframe_resize),evalkit_modal_iframe_resize(),document.getElementById("ek-modal-close").style.display="",document.getElementById("ek-modal-iframe-loading").contentWindow.document.write('<html><head></head><body style="text-align:center;"><img src="'+evalkit_setup.account_url+'/Media/Images/loadingd2l.gif" style="padding-top:10%;" /></body></html>')):a.innerHTML=e.body,void 0!==e.title&&null!==e.title?(i.innerHTML=e.title,i.style.display=""):i.style.display="none",s.setAttribute("class","ek-overlay-"+e.type),"blocker"===e.type?(0===(t=window.getComputedStyle(document.getElementById("header"),null).getPropertyValue("background-color")).length&&(t="rgb(57, 75, 88)"),s.style.background=t):s.style.background="",o.setAttribute("oncontextmenu","return false"),s.setAttribute("oncontextmenu","return false"),s.style.display="",o.style.display="",o.setAttribute("data-type",e.type),window.scrollTo(0,0),$("body").addClass("ek-modal-open"),setTimeout(function(){$("body").removeClass("ek-modal-open")},2e3),void 0!==e.buttons&&null!==e.buttons&&0<e.buttons.length){l.innerHTML="";for(var n=0;n<e.buttons.length;n++)0===n&&e.buttons[n].focus(),l.appendChild(e.buttons[n]);l.style.display=""}else l.innerHTML="",l.style.display="none";!0!==e.showClose&&(r.style.display="none"),!0===e.refreshOnClose?r.onclick=function(e){$(document).keypress(function(e){if(27===e.keyCode)return e.preventDefault(),!1}),ek_modal.close(),document.location.href=document.location.href}:r.onclick=function(e){ek_modal.close()}}},e.close=function(){$("body").removeClass("ek-modal-open"),s.style.display="none",o.style.display="none",a.innerHTML=""},document.body.appendChild(s),document.body.appendChild(o),e}();function evalkit_modal_iframe(e){var t=document.getElementById("ek-modal-iframe"),n=document.getElementById("ek-modal-iframe-loading");t.style.display="",n.style.display="none"}function evalkit_modal_iframe_resize(){var e=document.getElementById("ek-modal-iframe");null!==e&&e.setAttribute("height",window.innerHeight-180);e=document.getElementById("ek-modal-iframe-loading");null!==e&&e.setAttribute("height",window.innerHeight-180)}function evalkit_createCookie(e,t){var n=new Date,n=Date.UTC(n.getFullYear(),n.getMonth(),n.getDate(),n.getHours(),n.getMinutes(),n.getSeconds(),n.getMilliseconds());sessionStorage.setItem(e+"_E",n),sessionStorage.setItem(e,t)}function evalkit_readCookie(e){var t=sessionStorage.getItem(e+"_E");if(null===t)return sessionStorage.removeItem(e),null;var n=new Date;return 18e5<Date.UTC(n.getFullYear(),n.getMonth(),n.getDate(),n.getHours(),n.getMinutes(),n.getSeconds(),n.getMilliseconds())-t?(sessionStorage.removeItem(e+"_E"),sessionStorage.removeItem(e),null):sessionStorage.getItem(e)}