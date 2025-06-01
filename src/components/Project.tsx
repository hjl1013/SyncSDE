import './Project.css'

import DescriptionIcon from '@mui/icons-material/Description';
import GitHubIcon from '@mui/icons-material/GitHub';


const Project = () => {
    return (
        <div className='project'>
            <div className='project__hero'>
                <div className='project__title'>
                    <p className='google-sans-semi-bold'>SyncSDE: A Probabilistic Framework for Diffusion Synchronization</p>
                </div>
                <div className='project__authorInfo'>
                    <div className='project__authors'>
                        <div className='project__author'><p className='project__authorName google-sans-light'><a href='https://hjl1013.github.io/' target='_blank' rel='noopener noreferrer'>Hyunjun Lee</a></p><p className='google-sans-light'><sup>1&#8251;</sup></p></div>
                        <div className='project__author'><p className='project__authorName google-sans-light'><a href='https://www.linkedin.com/in/philip21' target='_blank' rel='noopener noreferrer'>Hyunsoo Lee</a></p><p className='google-sans-light'><sup>1&#8251;</sup></p></div>
                        <div className='project__author'><p className='project__authorName google-sans-light'><a href='https://jellyheadandrew.github.io/' target='_blank' rel='noopener noreferrer'>Sookwan Han</a></p><p className='google-sans-light'><sup>1,2&dagger;</sup></p></div>
                    </div>
                    <div className='project__affilations'>
                        <p className='project__affiliation google-sans-light'><sup>1</sup>ECE, Seoul National University</p>
                        <p className='project__affiliation google-sans-light'><sup>2</sup>Republic of Korea Air Force</p>
                    </div>
                    <div className='project__contributions'>
                        <p className='google-sans-light'>&#8251; indicates equal contribution</p>
                        <p className='google-sans-light'>&dagger; indicates project lead</p>
                    </div>
                    <div className='project__conference'>
                        <p className='project__conference google-sans-medium'>CVPR 2025</p>
                    </div>
                </div>
                <div className='project__links'>
                    <a href='https://arxiv.org/abs/2503.21555' target='_blank' rel='noopener noreferrer'>
                        <div className='project__link'>
                            <DescriptionIcon />
                            <p className='project__linkName google-sans-medium'>Paper</p>
                        </div>
                    </a>
                    <a href='https://github.com/hjl1013/SyncSDE' target='_blank' rel='noopener noreferrer'>
                        <div className='project__link'>
                            <GitHubIcon />
                            <p className='project__linkName google-sans-medium'>Code</p>
                        </div>
                    </a>
                    {/* <div className='project__link'>
                        <GitHubIcon />
                        <p className='project__linkName google-sans-medium'>Code (Coming Soon)</p>
                    </div> */}
                </div>
            </div>

            <div className='project__body'>
                <div className='project__teaser project__content'>
                    <img src='./figures/CVPR2025_Qualitative_Teaser_Crop.png' width='100%' height='100%'/>
                    <p className='project__teaserText google-sans-light'>
                        SyncSDE analyzes diffusion synchronization to identify where the correlation strategies should be focused, 
                        enabling coherent and high-quality results across diverse collaborative generation tasks.
                    </p>
                </div>
                <div className='project__abstractContainer grey-background'>
                    <div className='project__abstract project__content'>
                        <p className='project__bodyTitle google-sans-medium'>Abstract</p>
                        <p className='project__abstractText google-sans-light'>
                            There have been many attempts to leverage multiple diffusion models for collaborative generation, 
                            extending beyond the original domain. A prominent approach involves synchronizing multiple diffusion 
                            trajectories by mixing the estimated scores to artificially correlate the generation processes. However, 
                            existing methods rely on naive heuristics, such as averaging, without considering task specificity. 
                            These approaches do not clarify why such methods work and often fail when a heuristic suitable for one 
                            task is blindly applied to others. In this paper, we present a probabilistic framework for analyzing 
                            why diffusion synchronization works and reveal where heuristics should be focused—modeling correlations 
                            between multiple trajectories and adapting them to each specific task. We further identify optimal correlation 
                            models per task, achieving better results than previous approaches that apply a single heuristic across 
                            all tasks without justification.
                        </p>
                    </div>
                </div>
                <div className='project__experiments project__content'>
                    <div className='project__experiment'>
                        <p className='project__bodyTitle google-sans-medium'>Mask-based Text-to-Image Generation</p>
                        <div className='project__maskBody'>
                            <img src='./figures/CVPR2025_Qualitative_Mask_Generation.png' width='100%' height='100%'/>
                            <p className='google-sans-light'>
                                We first show mask-based text-to-image generation, generating masked region and background given seperate text prompts.
                            </p>
                        </div>
                    </div>
                    <div className='project__experiment'>
                        <p className='project__bodyTitle google-sans-medium'>Text-driven Real Image Editing</p>
                        <div className='project__editingBody'>
                            <p className='google-sans-light'>
                                Editing images given a text prompt to modify the content.
                            </p>
                            <img src='./figures/CVPR2025_Qualitative_Image_Editing_Crop.png' width='100%' height='100%'/>
                        </div>
                    </div>
                    <div className='project__experiment'>
                        <p className='project__bodyTitle google-sans-medium'>Wide Image Generation</p>
                        <div className='project__wideBody'>
                            <img src='./figures/CVPR2025_Qualitative_Wide_Image_Crop.png' width='100%' height='100%'/>
                            <p className='google-sans-light'>
                                Generating images with high resolution that a single diffusion model cannot generate.
                            </p>
                        </div>
                    </div>
                    <div className='project__experiment'>
                        <p className='project__bodyTitle google-sans-medium'>Ambiguous Image Generation</p>
                        <div className='project__ambiguousBody'>
                            <p className='google-sans-light'>
                                Generating images with optical illusions. It has different interpretations depending on the viewpoint and transformation.
                            </p>
                            <img src='./figures/CVPR2025_Qualitative_Ambiguous_Images.png' width='100%' height='100%'/>
                        </div>
                    </div>
                    <div className='project__experiment'>
                        <p className='project__bodyTitle google-sans-medium'>3D Mesh Texturing</p>
                        <div className='project__meshBody'>
                            <img src='./figures/CVPR2025_Qualitative_Mesh_Texturing.png' width='100%' height='100%'/>
                            <p className='google-sans-light'>
                                Generating the texture of 3D meshes given a text prompt.
                            </p>
                        </div>
                    </div>
                    <div className='project__experiment'>
                        <p className='project__bodyTitle google-sans-medium'>Long-horizon Motion Generation</p>
                        <div className='project__motionBody'>
                            <p className='google-sans-light'>
                                Using motion diffusion models to generate long-horizon motion with multiple motion prompts.
                            </p>
                            <img src='./figures/CVPR2025_Qualizative_Motion_Generation.png' width='100%' height='100%'/>
                        </div>
                    </div>
                </div>

                <div className='project__lambdaTesting project__content'>
                    <p className='project__bodyTitle google-sans-medium'>Lambda Testing</p>
                    <div className='project__lambdaTestingBody'>
                        <p className='google-sans-light'>
                            We experiment the effect of lambda, the hyperparameter that controls the collaboration between multiple diffusion trajectories.
                            As lambda decreases, it shows good collaboration between two trajectories.
                        </p>
                        <img src='./figures/CVPR2025_lambda_testing.png' width='100%' height='100%'/>
                    </div>
                </div>
                
                <div className='project__revisionReportContainer grey-background'>
                    <div className='project__revisionReport project__content'>
                        <p className='project__bodyTitle google-sans-medium'>Implementation Revisions and Updates</p>
                        <div className='project__revisionReportBody'>
                            <img src='./figures/CVPR2025_Revision_Report.svg' width='100%' height='100%'/>
                            <p className='google-sans-light'>
                                We identified some implementation errors for 3D Mesh Texturing and have updated the results in our official camera-ready version of CVPR.<br/><br/>

                                Our intention was to define the background mask as the region already filled by previous views during the autoregressive texture generation process. However, we mistakenly used a static background mask as shown in the figure on the left. <br/><br/>

                                Despite these changes, the overall tendency of the results remains consistent. Detailed explanation related to the revision is provided in our latest arXived paper section A.5.
                            </p>
                        </div>
                    </div>
                </div>

                <div className='project__bibtex project__content'>
                    <p className='project__bodyTitle google-sans-medium'>BibTeX</p>
                    <pre className='google-sans-light grey-background'>
                        <code>
                            {`@article{lee2025syncsde,\n\ttitle={SyncSDE: A Probabilistic Framework for Diffusion Synchronization},\n\tauthor={Lee, Hyunjun and Lee, Hyunsoo and Han, Sookwan},\n\tbooktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},\n\tyear={2025}\n}`}
                        </code>
                    </pre>
                </div>
            </div>
        </div>
    )
}

export default Project